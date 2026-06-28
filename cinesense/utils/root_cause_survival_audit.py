import os
import sys
import json
import numpy as np
import pandas as pd

# Set PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cinesense.utils.model_storage import load_model
from cinesense.services.recommendation import RecommendationService, get_franchise
from cinesense.retrieval.hybrid_c import top_retrieval_indices, hybrid_c_retrieval_scores
from cinesense.ranking.weighted_b import weighted_max_similarity_to_train_items, rerank_candidates

def find_anime_id_by_title(title, catalog_meta):
    title_low = title.lower().strip()
    for aid, meta in catalog_meta.items():
        if meta["title"].lower().strip() == title_low:
            return aid
        if meta.get("title_english") and meta["title_english"].lower().strip() == title_low:
            return aid
    # Fallback: partial match
    for aid, meta in catalog_meta.items():
        if title_low in meta["title"].lower():
            return aid
        if meta.get("title_english") and title_low in meta["title_english"].lower():
            return aid
    return None

def main():
    print("Loading model and gold standard dataset...", flush=True)
    model, catalog_df, metadata = load_model(os.path.join(PROJECT_ROOT, "cinesense/models/twostage_v1"))
    service = RecommendationService(model, catalog_df)

    # Enable O(1) franchise root caching
    original_get_franchise_root = service.get_franchise_root
    franchise_root_cache = {}
    def cached_get_franchise_root(franchise_name):
        if franchise_name not in franchise_root_cache:
            franchise_root_cache[franchise_name] = original_get_franchise_root(franchise_name)
        return franchise_root_cache[franchise_name]
    service.get_franchise_root = cached_get_franchise_root

    # Load gold standard dataset
    with open(os.path.join(PROJECT_ROOT, "evaluation/gold_standard_v2.json"), "r") as f:
        gold_dataset = json.load(f)

    # Focus seeds
    target_seed_names = [
        "Hunter x Hunter",
        "One Piece",
        "Naruto",
        "Fullmetal Alchemist: Brotherhood",
        "Fate/Zero",
        "Black Lagoon",
        "Mushishi",
        "Death Note"
    ]

    # Map target seeds
    target_entries = []
    for name in target_seed_names:
        for entry in gold_dataset:
            if entry["seed"].lower().strip() == name.lower().strip() or \
               (name == "Fullmetal Alchemist: Brotherhood" and entry["seed"] == "FMAB") or \
               (name == "Hunter x Hunter" and entry["seed"] == "Hunter x Hunter") or \
               (name == "Death Note" and entry["seed"] == "Death Note"):
                target_entries.append(entry)
                break

    # Resolve target recommendations to IDs
    resolved_targets = {}
    for entry in target_entries:
        seed_name = entry["seed"]
        seed_id = entry["anime_id"]
        
        good_ids = []
        for title in entry["good_recommendations"]:
            aid = find_anime_id_by_title(title, service.catalog_meta)
            if aid is not None: good_ids.append(aid)
            else: print(f"Warning: Could not resolve title '{title}' for seed '{seed_name}'")
            
        acceptable_ids = []
        for title in entry["acceptable_recommendations"]:
            aid = find_anime_id_by_title(title, service.catalog_meta)
            if aid is not None: acceptable_ids.append(aid)
            else: print(f"Warning: Could not resolve title '{title}' for seed '{seed_name}'")
            
        resolved_targets[seed_id] = {
            "name": seed_name,
            "good": set(good_ids),
            "acceptable": set(acceptable_ids),
            "relevant": set(good_ids + acceptable_ids)
        }

    # Force baseline production settings
    os.environ["CINESENSE_REPRESENTATION_PENALTY"] = "True"
    os.environ["CINESENSE_REPRESENTATION_LAMBDA"] = "0.03"

    coverage_stats = []
    survival_counts = []
    failure_attribution = {}

    for seed_id, info in resolved_targets.items():
        seed_name = info["name"]
        good_set = info["good"]
        relevant_set = info["relevant"]
        
        # --- Stage 1: Embedding Neighbors ---
        idx_s = model.item_id_to_index[seed_id]
        emb_s = model.catalog_embeddings[idx_s]
        
        sims = []
        for i in range(len(model.catalog_embeddings)):
            if i == idx_s:
                continue
            sim = float(np.dot(emb_s, model.catalog_embeddings[i]))
            sims.append((int(model.anime_ids[i]), sim))
        sims.sort(key=lambda x: -x[1])
        
        # Check coverage
        neighbors_50 = {item_id for item_id, _ in sims[:50]}
        neighbors_100 = {item_id for item_id, _ in sims[:100]}
        neighbors_300 = {item_id for item_id, _ in sims[:300]}
        neighbors_500 = {item_id for item_id, _ in sims[:500]}
        
        coverage_stats.append({
            "seed": seed_name,
            "good_50": len(good_set.intersection(neighbors_50)),
            "good_100": len(good_set.intersection(neighbors_100)),
            "good_300": len(good_set.intersection(neighbors_300)),
            "good_500": len(good_set.intersection(neighbors_500)),
            "relevant_50": len(relevant_set.intersection(neighbors_50)),
            "relevant_100": len(relevant_set.intersection(neighbors_100)),
            "relevant_300": len(relevant_set.intersection(neighbors_300)),
            "relevant_500": len(relevant_set.intersection(neighbors_500)),
            "total_relevant": len(relevant_set)
        })

        # --- Stage 2: Pipeline Survival Analysis ---
        # We trace exactly which relevant items survive each stage of the discover pipeline
        
        # 1. Retrieval
        retrieval_k = 300
        train_indices = np.asarray([idx_s], dtype=np.int32)
        train_items = {seed_id}
        
        retrieval_scores = hybrid_c_retrieval_scores(
            train_indices,
            model.catalog_embeddings,
            model.popularity_scores,
            model.semantic_weight,
            model.popularity_weight,
            model.seed_batch_size,
        )
        retrieved_indices_raw = top_retrieval_indices(
            retrieval_scores,
            train_items,
            model.anime_ids,
            retrieval_k,
        )
        retrieved_ids = {int(model.anime_ids[idx]) for idx in retrieved_indices_raw}
        
        # 2. Retrieval Franchise Exclusion & Truncation to 150
        seed_franchises = set()
        meta_s = service.catalog_meta.get(seed_id)
        if meta_s:
            seed_franchises.add(get_franchise(meta_s["title"]))
            if meta_s.get("title_english"):
                seed_franchises.add(get_franchise(meta_s["title_english"]))
                
        retrieved_indices_prepared = []
        for r_idx in retrieved_indices_raw:
            anime_id = int(model.anime_ids[r_idx])
            meta = service.catalog_meta[anime_id]
            title = meta["title"]
            eng_title = meta.get("title_english") or ""
            cand_f = get_franchise(title)
            cand_f_eng = get_franchise(eng_title) if eng_title else ""
            
            if cand_f in seed_franchises or (cand_f_eng and cand_f_eng in seed_franchises):
                continue
                
            retrieved_indices_prepared.append(r_idx)
            if len(retrieved_indices_prepared) == 150:
                break
        ranked_pool_ids = {int(model.anime_ids[idx]) for idx in retrieved_indices_prepared}
        
        # 3. Ranking Stage (greedy selection)
        train_weights = np.asarray([1.0], dtype=np.float32)
        weighted_semantic_scores = weighted_max_similarity_to_train_items(
            train_indices,
            train_weights,
            model.catalog_embeddings,
            model.seed_batch_size,
        )
        rerank_scores = (
            model.semantic_weight * weighted_semantic_scores
            + model.popularity_weight * model.popularity_scores
        )
        recommendations = rerank_candidates(
            np.asarray(retrieved_indices_prepared, dtype=np.int32),
            rerank_scores,
            retrieval_scores,
            model.anime_ids,
            150,
            representation_penalty=True,
            representation_lambda=0.03,
            train_indices=train_indices,
            catalog_embeddings=model.catalog_embeddings,
        )
        # Sort scores to map enriched
        scores = {rec_id: float(rerank_scores[model.item_id_to_index[rec_id]]) for rec_id in recommendations}
        enriched = service.enrich_recommendations(recommendations, scores, [seed_id], weighted_semantic_scores)
        
        # 4. Discover Filtering
        filtered_enriched = []
        seen_rec_franchises = set()
        for item in enriched:
            rec_id = item["anime_id"]
            rec_title = item["title"]
            rec_eng_title = item.get("title_english")
            
            rec_f_name = get_franchise(rec_title)
            rec_f_eng_name = get_franchise(rec_eng_title) if rec_eng_title else ""
            
            if rec_f_name in seed_franchises or (rec_f_eng_name and rec_f_eng_name in seed_franchises):
                continue
                
            root_id = service.get_franchise_root(rec_f_name)
            is_sequel = False
            if root_id is not None and rec_id != root_id:
                is_sequel = True
            elif service.is_sequel_title(rec_title) or (rec_eng_title and service.is_sequel_title(rec_eng_title)):
                is_sequel = True
                
            if is_sequel:
                continue
                
            if rec_f_name in seen_rec_franchises or (rec_f_eng_name and rec_f_eng_name in seen_rec_franchises):
                continue
                
            filtered_enriched.append(item)
            seen_rec_franchises.add(rec_f_name)
            if rec_f_eng_name:
                seen_rec_franchises.add(rec_f_eng_name)
                
        survived_filter_ids = {item["anime_id"] for item in filtered_enriched}
        final_top10_ids = {item["anime_id"] for item in filtered_enriched[:10]}

        # Trace each relevant item
        failures = {
            "embedding": 0,
            "retrieval": 0,
            "ranking": 0,
            "filtering": 0
        }
        
        # Track items that survived
        survived_count = 0
        
        for item_id in relevant_set:
            if item_id in final_top10_ids:
                survived_count += 1
                continue
            
            # Map failure stage
            if item_id not in neighbors_500:
                failures["embedding"] += 1
            elif item_id not in ranked_pool_ids:
                failures["retrieval"] += 1
            elif item_id not in survived_filter_ids:
                failures["filtering"] += 1
            else:
                failures["ranking"] += 1

        total_failures = sum(failures.values())
        if total_failures > 0:
            failure_pcts = {k: (v / total_failures) * 100.0 for k, v in failures.items()}
        else:
            failure_pcts = {k: 0.0 for k in failures.keys()}
            
        failure_attribution[seed_id] = failure_pcts
        
        survival_counts.append({
            "seed": seed_name,
            "relevant": len(relevant_set),
            "in_neighbors_500": len(relevant_set.intersection(neighbors_500)),
            "in_ranked_pool": len(relevant_set.intersection(ranked_pool_ids)),
            "survived_filters": len(relevant_set.intersection(survived_filter_ids)),
            "final_top10": len(relevant_set.intersection(final_top10_ids))
        })

    # --- PRINT DELIVERABLES ---
    print("\n" + "="*80)
    print("STEP 1: GOLD STANDARD COVERAGE TABLES")
    print("="*80)
    print("Good Recommendations Coverage:")
    print(f"| {'Seed':<40} | {'Good@50':<8} | {'Good@100':<8} | {'Good@300':<8} | {'Good@500':<8} |")
    print(f"| {'-'*40} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} |")
    for r in coverage_stats:
        print(f"| {r['seed']:<40} | {r['good_50']:<8} | {r['good_100']:<8} | {r['good_300']:<8} | {r['good_500']:<8} |")
    print()
    
    print("Relevant Recommendations (Good + Acceptable) Coverage:")
    print(f"| {'Seed':<40} | {'Rel@50':<8} | {'Rel@100':<8} | {'Rel@300':<8} | {'Rel@500':<8} | {'Total':<5} |")
    print(f"| {'-'*40} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*5} |")
    for r in coverage_stats:
        print(f"| {r['seed']:<40} | {r['relevant_50']:<8} | {r['relevant_100']:<8} | {r['relevant_300']:<8} | {r['relevant_500']:<8} | {r['total_relevant']:<5} |")
    print()

    print("="*80)
    print("STEP 2: PIPELINE SURVIVAL TABLES")
    print("="*80)
    print(f"| {'Seed':<40} | {'Relevant':<8} | {'Neigh_500':<9} | {'Rank_Pool':<9} | {'Filter_Surv':<11} | {'Final_Top10':<11} |")
    print(f"| {'-'*40} | {'-'*8} | {'-'*9} | {'-'*9} | {'-'*11} | {'-'*11} |")
    for r in survival_counts:
        print(f"| {r['seed']:<40} | {r['relevant']:<8} | {r['in_neighbors_500']:<9} | {r['in_ranked_pool']:<9} | {r['survived_filters']:<11} | {r['final_top10']:<11} |")
    print()

    print("="*80)
    print("STEP 3: FAILURE ATTRIBUTION MATRIX")
    print("="*80)
    print(f"| {'Seed':<40} | {'Embedding':<9} | {'Retrieval':<9} | {'Ranking':<9} | {'Filtering':<9} |")
    print(f"| {'-'*40} | {'-'*9} | {'-'*9} | {'-'*9} | {'-'*9} |")
    for seed_id, info in resolved_targets.items():
        seed_name = info["name"]
        pcts = failure_attribution[seed_id]
        print(f"| {seed_name:<40} | {pcts['embedding']:<8.1f}% | {pcts['retrieval']:<8.1f}% | {pcts['ranking']:<8.1f}% | {pcts['filtering']:<8.1f}% |")
    print()

    print("="*80)
    print("STEP 4: GLOBAL AGGREGATION")
    print("="*80)
    
    # Calculate global sums of counts
    global_failures = {
        "embedding": 0,
        "retrieval": 0,
        "ranking": 0,
        "filtering": 0
    }
    
    total_relevant_global = 0
    total_survived_global = 0
    
    for seed_id, info in resolved_targets.items():
        seed_name = info["name"]
        # get survival records
        sc = next(x for x in survival_counts if x["seed"] == seed_name)
        cov = next(x for x in coverage_stats if x["seed"] == seed_name)
        
        # trace failures
        rel_set = info["relevant"]
        total_relevant_global += len(rel_set)
        
        # final top 10
        final_top10_ids = next(x for x in survival_counts if x["seed"] == seed_name)["final_top10"]
        total_survived_global += final_top10_ids
        
        # Get neighbors and trace failures manually for global aggregate
        # Repeat classification to avoid float rounding errors
        idx_s = model.item_id_to_index[seed_id]
        emb_s = model.catalog_embeddings[idx_s]
        sims = []
        for i in range(len(model.catalog_embeddings)):
            if i == idx_s: continue
            sim = float(np.dot(emb_s, model.catalog_embeddings[i]))
            sims.append((int(model.anime_ids[i]), sim))
        sims.sort(key=lambda x: -x[1])
        neighbors_500 = {item_id for item_id, _ in sims[:500]}
        
        # Run recommend stages
        res = run_custom_recommend_ids(service, seed_id, model)
        ranked_pool_ids = res["ranked_ids"]
        survived_filter_ids = res["survived_filter_ids"]
        final_ids = res["final_ids"]
        
        for item_id in rel_set:
            if item_id in final_ids:
                continue
            if item_id not in neighbors_500:
                global_failures["embedding"] += 1
            elif item_id not in ranked_pool_ids:
                global_failures["retrieval"] += 1
            elif item_id not in survived_filter_ids:
                global_failures["filtering"] += 1
            else:
                global_failures["ranking"] += 1

    total_failures_global = sum(global_failures.values())
    print(f"Total Evaluated Recommendations: {total_relevant_global}")
    print(f"Total Successfully Recommended:  {total_survived_global}")
    print(f"Total Missing Recommendations:   {total_failures_global}")
    print()
    print(f"| {'Failure Source':<25} | {'Count':<5} | {'Contribution %':<15} |")
    print(f"| {'-'*25} | {'-'*5} | {'-'*15} |")
    for k, v in global_failures.items():
        name_map = {
            "embedding": "Embedding Quality",
            "retrieval": "Retrieval Suppression",
            "ranking": "Ranking Suppression",
            "filtering": "Filtering"
        }
        pct = (v / total_failures_global) * 100.0 if total_failures_global > 0 else 0.0
        print(f"| {name_map[k]:<25} | {v:<5} | {pct:<14.2f}% |")
    print()

def run_custom_recommend_ids(service, seed_id, model):
    valid_ids, validated_ratings = service.validate_inputs([seed_id], {seed_id: 10.0}, 10)
    train_indices = np.asarray([model.item_id_to_index[aid] for aid in valid_ids], dtype=np.int32)
    train_items = set(valid_ids)
    
    retrieval_k = 300
    retrieval_scores = hybrid_c_retrieval_scores(
        train_indices,
        model.catalog_embeddings,
        model.popularity_scores,
        model.semantic_weight,
        model.popularity_weight,
        model.seed_batch_size,
    )
    retrieved_indices_raw = top_retrieval_indices(
        retrieval_scores,
        train_items,
        model.anime_ids,
        retrieval_k,
    )
    
    seed_franchises = set()
    for aid in valid_ids:
        meta = service.catalog_meta.get(aid)
        if meta:
            seed_franchises.add(get_franchise(meta["title"]))
            if meta.get("title_english"):
                seed_franchises.add(get_franchise(meta["title_english"]))
                
    retrieved_indices_prepared = []
    for r_idx in retrieved_indices_raw:
        anime_id = int(model.anime_ids[r_idx])
        meta = service.catalog_meta[anime_id]
        title = meta["title"]
        eng_title = meta.get("title_english") or ""
        cand_f = get_franchise(title)
        cand_f_eng = get_franchise(eng_title) if eng_title else ""
        
        if cand_f in seed_franchises or (cand_f_eng and cand_f_eng in seed_franchises):
            continue
            
        retrieved_indices_prepared.append(r_idx)
        if len(retrieved_indices_prepared) == 150:
            break
            
    retrieved_indices = np.asarray(retrieved_indices_prepared, dtype=np.int32)
    train_weights = np.asarray([1.0], dtype=np.float32)
    
    weighted_semantic_scores = weighted_max_similarity_to_train_items(
        train_indices,
        train_weights,
        model.catalog_embeddings,
        model.seed_batch_size,
    )
    rerank_scores = (
        model.semantic_weight * weighted_semantic_scores
        + model.popularity_weight * model.popularity_scores
    )
    
    recommendations = rerank_candidates(
        retrieved_indices,
        rerank_scores,
        retrieval_scores,
        model.anime_ids,
        150,
        representation_penalty=True,
        representation_lambda=0.03,
        train_indices=train_indices,
        catalog_embeddings=model.catalog_embeddings,
    )
    
    scores = {rec_id: float(rerank_scores[model.item_id_to_index[rec_id]]) for rec_id in recommendations}
    enriched = service.enrich_recommendations(recommendations, scores, valid_ids, weighted_semantic_scores)
    
    filtered_enriched = []
    seen_rec_franchises = set()
    for item in enriched:
        rec_id = item["anime_id"]
        rec_title = item["title"]
        rec_eng_title = item.get("title_english")
        
        rec_f_name = get_franchise(rec_title)
        rec_f_eng_name = get_franchise(rec_eng_title) if rec_eng_title else ""
        
        if rec_f_name in seed_franchises or (rec_f_eng_name and rec_f_eng_name in seed_franchises):
            continue
            
        root_id = service.get_franchise_root(rec_f_name)
        is_sequel = False
        if root_id is not None and rec_id != root_id:
            is_sequel = True
        elif service.is_sequel_title(rec_title) or (rec_eng_title and service.is_sequel_title(rec_eng_title)):
            is_sequel = True
            
        if is_sequel:
            continue
            
        if rec_f_name in seen_rec_franchises or (rec_f_eng_name and rec_f_eng_name in seen_rec_franchises):
            continue
            
        filtered_enriched.append(item)
        seen_rec_franchises.add(rec_f_name)
        if rec_f_eng_name:
            seen_rec_franchises.add(rec_f_eng_name)
            
    return {
        "ranked_ids": {int(model.anime_ids[idx]) for idx in retrieved_indices_prepared},
        "survived_filter_ids": {item["anime_id"] for item in filtered_enriched},
        "final_ids": {item["anime_id"] for item in filtered_enriched[:10]}
    }

if __name__ == "__main__":
    main()
