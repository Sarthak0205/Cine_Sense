import os
import sys
import re
import numpy as np
import pandas as pd

# Set PYTHONPATH
PROJECT_ROOT = "/Users/sdc/Projects/CineSense-main"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cinesense.utils.model_storage import load_model
from cinesense.services.recommendation import RecommendationService, get_franchise
from cinesense.retrieval.hybrid_c import hybrid_c_retrieval_scores, top_retrieval_indices
from cinesense.ranking.weighted_b import weighted_max_similarity_to_train_items, rerank_candidates

def main():
    print("Loading model for FMAB neighborhood audit...", flush=True)
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

    # Helper to calculate seed shares
    def get_seed_shares(recs, seeds):
        seed_embeddings = {s_id: model.catalog_embeddings[model.item_id_to_index[s_id]] for s_id in seeds}
        counts = {s_id: 0 for s_id in seeds}
        for item in recs:
            emb_c = model.catalog_embeddings[model.item_id_to_index[item["anime_id"]]]
            best_seed = None
            max_sim = -float('inf')
            for s_id in seeds:
                sim = float(np.dot(emb_c, seed_embeddings[s_id]))
                if sim > max_sim:
                    max_sim = sim
                    best_seed = s_id
            if best_seed in counts:
                counts[best_seed] += 1
        shares = {}
        for s_id, count in counts.items():
            shares[seeds_config.get(s_id, f"Id{s_id}")] = count / len(recs) if recs else 0.0
        return shares

    # Seeds mapping
    seeds_config = {
        5114: "FMAB",
        1535: "DN",
        9253: "SG",
        11061: "HxH"
    }
    
    # ----------------------------------------------------
    # Part 1 — Embedding Neighborhood Analysis
    # ----------------------------------------------------
    print("\n" + "="*80)
    print("PART 1: EMBEDDING NEIGHBORHOOD ANALYSIS")
    print("="*80)

    for sid, name in seeds_config.items():
        print(f"\n### Neighborhood for seed: {name} (ID: {sid})")
        idx_s = model.item_id_to_index[sid]
        emb_s = model.catalog_embeddings[idx_s]
        
        # Calculate raw similarities to all items in catalog (excluding the seed itself)
        sims = []
        for i in range(len(model.catalog_embeddings)):
            if i == idx_s:
                continue
            sim = float(np.dot(emb_s, model.catalog_embeddings[i]))
            sims.append((int(model.anime_ids[i]), sim, i))
            
        sims.sort(key=lambda x: -x[1])
        
        for k in [100, 300, 500]:
            sub = sims[:k]
            sim_vals = [x[1] for x in sub]
            
            avg_sim = np.mean(sim_vals)
            med_sim = np.median(sim_vals)
            std_sim = np.std(sim_vals)
            
            # Scores (semantic_weight * similarity + popularity_weight * popularity)
            score_vals = []
            for item_id, sim, i in sub:
                score = float(model.semantic_weight * sim + model.popularity_weight * model.popularity_scores[i])
                score_vals.append(score)
                
            avg_score = np.mean(score_vals)
            med_score = np.median(score_vals)
            std_score = np.std(score_vals)
            
            # Histogram ranges
            h_07 = sum(1 for v in sim_vals if v >= 0.7)
            h_06 = sum(1 for v in sim_vals if 0.6 <= v < 0.7)
            h_05 = sum(1 for v in sim_vals if 0.5 <= v < 0.6)
            h_04 = sum(1 for v in sim_vals if 0.4 <= v < 0.5)
            h_lt04 = sum(1 for v in sim_vals if v < 0.4)
            
            print(f"  **Top {k} Neighbors:**")
            print(f"    - Similarity: Avg={avg_sim:.4f}, Median={med_sim:.4f}, Std={std_sim:.4f}")
            print(f"    - Score:      Avg={avg_score:.4f}, Median={med_score:.4f}, Std={std_score:.4f}")
            print(f"    - Histogram:  >=0.7: {h_07} | 0.6-0.7: {h_06} | 0.5-0.6: {h_05} | 0.4-0.5: {h_04} | <0.4: {h_lt04}")
            
    # ----------------------------------------------------
    # Part 2 — Discoverability Analysis
    # ----------------------------------------------------
    print("\n" + "="*80)
    print("PART 2: DISCOVERABILITY ANALYSIS")
    print("="*80)

    keywords = ["movie", "film", "ova", "ona", "special", "specials", "recap", "pilot"]
    def is_movie_ova_special(t, eng_t):
        t_low = (t or "").lower()
        eng_low = (eng_t or "").lower()
        return any(kw in t_low or kw in eng_low for kw in keywords)

    for sid, name in seeds_config.items():
        print(f"\n### Filter breakdown for seed: {name}")
        idx_s = model.item_id_to_index[sid]
        emb_s = model.catalog_embeddings[idx_s]
        
        sims = []
        for i in range(len(model.catalog_embeddings)):
            if i == idx_s:
                continue
            sim = float(np.dot(emb_s, model.catalog_embeddings[i]))
            sims.append((int(model.anime_ids[i]), sim, i))
        sims.sort(key=lambda x: -x[1])

        # Get seed franchise
        seed_meta = service.catalog_meta[sid]
        seed_franchises = {get_franchise(seed_meta["title"])}
        if seed_meta.get("title_english"):
            seed_franchises.add(get_franchise(seed_meta["title_english"]))

        for k in [100, 300, 500]:
            sub = sims[:k]
            
            same_franchise = 0
            sequel = 0
            movie_ova = 0
            duplicate_franchise = 0
            discover_eligible = 0
            
            seen_franchises = set()
            
            for item_id, sim, i in sub:
                meta = service.catalog_meta[item_id]
                title = meta["title"]
                eng_title = meta.get("title_english") or ""
                
                # A. Same franchise check
                cand_f = get_franchise(title)
                cand_f_eng = get_franchise(eng_title) if eng_title else ""
                
                is_same_f = (cand_f in seed_franchises) or (cand_f_eng and cand_f_eng in seed_franchises)
                if is_same_f:
                    same_franchise += 1
                    continue
                    
                # B. Sequel check
                root_id = service.get_franchise_root(cand_f)
                is_seq = False
                if root_id is not None and item_id != root_id:
                    is_seq = True
                elif service.is_sequel_title(title) or (eng_title and service.is_sequel_title(eng_title)):
                    is_seq = True
                    
                if is_seq:
                    sequel += 1
                    continue
                    
                # C. Movie/OVA/Special check
                if is_movie_ova_special(title, eng_title):
                    movie_ova += 1
                    continue
                    
                # D. Duplicate check
                if cand_f in seen_franchises or (cand_f_eng and cand_f_eng in seen_franchises):
                    duplicate_franchise += 1
                else:
                    discover_eligible += 1
                    seen_franchises.add(cand_f)
                    if cand_f_eng:
                        seen_franchises.add(cand_f_eng)
                        
            print(f"  **Top {k} Neighbors:**")
            print(f"    - Same Franchise: {same_franchise:<4} | Sequel: {sequel:<4} | Movie/OVA: {movie_ova:<4} | Duplicate: {duplicate_franchise:<4} | Discover Eligible: {discover_eligible:<4}")

    # ----------------------------------------------------
    # Part 3 — Survival Analysis
    # ----------------------------------------------------
    print("\n" + "="*80)
    print("PART 3: PIPELINE SURVIVAL ANALYSIS")
    print("="*80)

    print(f"| {'Seed':<6} | {'Retrieved':<9} | {'After Excl':<10} | {'After Rank':<10} | {'After Sequel':<12} | {'After Dedup':<11} | {'Final Top 10':<12} |")
    print(f"| {'-'*6} | {'-'*9} | {'-'*10} | {'-'*10} | {'-'*12} | {'-'*11} | {'-'*12} |")

    for sid, name in seeds_config.items():
        # Retrieve 300 candidates
        train_indices = np.asarray([model.item_id_to_index[sid]], dtype=np.int32)
        train_items = {sid}
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
            300,
        )
        
        # Franchise Exclusion
        seed_meta = service.catalog_meta[sid]
        seed_franchises = {get_franchise(seed_meta["title"])}
        if seed_meta.get("title_english"):
            seed_franchises.add(get_franchise(seed_meta["title_english"]))
            
        retrieved_indices_prepared = []
        for idx in retrieved_indices_raw:
            anime_id = int(model.anime_ids[idx])
            meta = service.catalog_meta[anime_id]
            title = meta["title"]
            eng_title = meta.get("title_english") or ""
            cand_f = get_franchise(title)
            cand_f_eng = get_franchise(eng_title) if eng_title else ""
            if cand_f in seed_franchises or (cand_f_eng and cand_f_eng in seed_franchises):
                continue
            retrieved_indices_prepared.append(idx)
            
        # Ranking
        ranked_pool = retrieved_indices_prepared[:150]
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
            np.asarray(ranked_pool, dtype=np.int32),
            rerank_scores,
            retrieval_scores,
            model.anime_ids,
            150,
        )
        
        # Sequel Filter
        scores = {rid: float(rerank_scores[model.item_id_to_index[rid]]) for rid in recommendations}
        enriched = service.enrich_recommendations(recommendations, scores, [sid], weighted_semantic_scores)
        
        after_sequel = []
        for item in enriched:
            rec_id = item["anime_id"]
            rec_title = item["title"]
            rec_eng_title = item.get("title_english")
            rec_f_name = get_franchise(rec_title)
            
            root_id = service.get_franchise_root(rec_f_name)
            is_sequel = False
            if root_id is not None and rec_id != root_id:
                is_sequel = True
            elif service.is_sequel_title(rec_title) or (rec_eng_title and service.is_sequel_title(rec_eng_title)):
                is_sequel = True
                
            if is_sequel:
                continue
            after_sequel.append(item)
            
        # Franchise Deduplication
        after_dedup = []
        seen_rec_franchises = set()
        for item in after_sequel:
            rec_id = item["anime_id"]
            rec_title = item["title"]
            rec_eng_title = item.get("title_english")
            rec_f_name = get_franchise(rec_title)
            rec_f_eng_name = get_franchise(rec_eng_title) if rec_eng_title else ""
            
            if rec_f_name in seen_rec_franchises or (rec_f_eng_name and rec_f_eng_name in seen_rec_franchises):
                continue
            after_dedup.append(item)
            seen_rec_franchises.add(rec_f_name)
            if rec_f_eng_name:
                seen_rec_franchises.add(rec_f_eng_name)
                
        final_top10 = len(after_dedup[:10])
        print(f"| {name:<6} | {len(retrieved_indices_raw):<9} | {len(retrieved_indices_prepared):<10} | {len(ranked_pool):<10} | {len(after_sequel):<12} | {len(after_dedup):<11} | {final_top10:<12} |")
    print()

    # ----------------------------------------------------
    # Part 4 — Rank Analysis
    # ----------------------------------------------------
    print("\n" + "="*80)
    print("PART 4: RANK ANALYSIS")
    print("="*80)

    for sid, name in seeds_config.items():
        print(f"\n### Rank Analysis for seed: {name}")
        idx_s = model.item_id_to_index[sid]
        emb_s = model.catalog_embeddings[idx_s]
        
        sims = []
        for i in range(len(model.catalog_embeddings)):
            if i == idx_s:
                continue
            sim = float(np.dot(emb_s, model.catalog_embeddings[i]))
            sims.append((int(model.anime_ids[i]), sim, i))
        sims.sort(key=lambda x: -x[1])
        
        seed_meta = service.catalog_meta[sid]
        seed_franchises = {get_franchise(seed_meta["title"])}
        if seed_meta.get("title_english"):
            seed_franchises.add(get_franchise(seed_meta["title_english"]))
            
        first_disc_rank_raw = -1
        first_disc_rank_ranked = -1
        
        seen_franchises = set()
        
        # Let's rebuild the discover eligibility list sequentially for raw list
        discover_eligible_items = []
        for rank_idx, (item_id, sim, i) in enumerate(sims):
            meta = service.catalog_meta[item_id]
            title = meta["title"]
            eng_title = meta.get("title_english") or ""
            
            # Exclusion checks
            cand_f = get_franchise(title)
            cand_f_eng = get_franchise(eng_title) if eng_title else ""
            if cand_f in seed_franchises or (cand_f_eng and cand_f_eng in seed_franchises):
                continue
            root_id = service.get_franchise_root(cand_f)
            is_seq = False
            if root_id is not None and item_id != root_id:
                is_seq = True
            elif service.is_sequel_title(title) or (eng_title and service.is_sequel_title(eng_title)):
                is_seq = True
            if is_seq:
                continue
            if is_movie_ova_special(title, eng_title):
                continue
            if cand_f in seen_franchises or (cand_f_eng and cand_f_eng in seen_franchises):
                continue
                
            seen_franchises.add(cand_f)
            discover_eligible_items.append((item_id, rank_idx + 1))
            if first_disc_rank_raw == -1:
                first_disc_rank_raw = rank_idx + 1
                
        # Now let's calculate counts in Top 50, 100, 300
        in_50 = sum(1 for item_id, rank in discover_eligible_items if rank <= 50)
        in_100 = sum(1 for item_id, rank in discover_eligible_items if rank <= 100)
        in_300 = sum(1 for item_id, rank in discover_eligible_items if rank <= 300)
        
        print(f"  - First discoverable rank in raw list: {first_disc_rank_raw}")
        print(f"  - Count of discoverable items in Top 50:  {in_50}")
        print(f"  - Count of discoverable items in Top 100: {in_100}")
        print(f"  - Count of discoverable items in Top 300: {in_300}")

    # ----------------------------------------------------
    # Part 5 — Multi-Seed Competition Analysis
    # ----------------------------------------------------
    print("\n" + "="*80)
    print("PART 5: MULTI-SEED COMPETITION ANALYSIS")
    print("="*80)

    multi_scenarios = [
        {"name": "FMAB + DN", "seeds": [5114, 1535]},
        {"name": "FMAB + SG", "seeds": [5114, 9253]},
        {"name": "FMAB + HxH", "seeds": [5114, 11061]},
        {"name": "FMAB + DN + SG", "seeds": [5114, 1535, 9253]},
        {"name": "FMAB + DN + AoT + HxH", "seeds": [5114, 1535, 16498, 11061]}
    ]

    for sc in multi_scenarios:
        name = sc["name"]
        seeds = sc["seeds"]
        ratings = {s: 10.0 - i for i, s in enumerate(seeds)}
        
        os.environ["CINESENSE_REPRESENTATION_PENALTY"] = "True"
        os.environ["CINESENSE_REPRESENTATION_LAMBDA"] = "0.03"
        
        # Warmup
        service.recommend(seeds, ratings=ratings, top_k=10, mode="discover")
        
        recs = service.recommend(seeds, ratings=ratings, top_k=10, mode="discover")
        avg_score = np.mean([r["score"] for r in recs]) if recs else 0.0
        
        # Attribution share (on retrieved / discoverable pool before slicing)
        # Reconstruct Stage 1 retrieval & discoverable candidates for seeds
        train_indices = np.asarray([model.item_id_to_index[aid] for aid in seeds], dtype=np.int32)
        train_weights = np.asarray([model._rating_weight(int(ratings[aid])) for aid in seeds], dtype=np.float32)
        retrieval_scores = hybrid_c_retrieval_scores(
            train_indices, model.catalog_embeddings, model.popularity_scores,
            model.semantic_weight, model.popularity_weight, model.seed_batch_size
        )
        retrieved_indices_raw = top_retrieval_indices(retrieval_scores, set(seeds), model.anime_ids, 300)
        
        # Exclude seed franchises
        seed_franchises = set()
        for aid in seeds:
            meta = service.catalog_meta.get(aid)
            if meta:
                seed_franchises.add(get_franchise(meta["title"]))
                if meta.get("title_english"):
                    seed_franchises.add(get_franchise(meta["title_english"]))
                    
        discover_eligible_pool = []
        seen_franchises = set()
        for idx in retrieved_indices_raw:
            anime_id = int(model.anime_ids[idx])
            meta = service.catalog_meta[anime_id]
            title = meta["title"]
            eng_title = meta.get("title_english") or ""
            
            cand_f = get_franchise(title)
            cand_f_eng = get_franchise(eng_title) if eng_title else ""
            if cand_f in seed_franchises or (cand_f_eng and cand_f_eng in seed_franchises):
                continue
            root_id = service.get_franchise_root(cand_f)
            is_seq = False
            if root_id is not None and anime_id != root_id:
                is_seq = True
            elif service.is_sequel_title(title) or (eng_title and service.is_sequel_title(eng_title)):
                is_seq = True
            if is_seq:
                continue
            if is_movie_ova_special(title, eng_title):
                continue
            if cand_f in seen_franchises or (cand_f_eng and cand_f_eng in seen_franchises):
                continue
            seen_franchises.add(cand_f)
            discover_eligible_pool.append(idx)
            
        # Attribute discover eligible pool to closest seed
        seed_embeddings = {s_id: model.catalog_embeddings[model.item_id_to_index[s_id]] for s_id in seeds}
        pool_attribs = {s_id: 0 for s_id in seeds}
        for idx in discover_eligible_pool:
            emb_c = model.catalog_embeddings[idx]
            best_seed = None
            max_sim = -float('inf')
            for s_id, emb_s in seed_embeddings.items():
                sim = float(np.dot(emb_c, emb_s))
                if sim > max_sim:
                    max_sim = sim
                    best_seed = s_id
            if best_seed in pool_attribs:
                pool_attribs[best_seed] += 1
                
        # Representation shares in final Top 10
        final_shares = get_seed_shares(recs, seeds)
        
        print(f"\n### Scenario: {name}")
        print(f"  - Average Recommendation Score: {avg_score:.4f}")
        
        print("  - Attribution Share (in discover-eligible candidate pool):")
        total_attrib = sum(pool_attribs.values())
        for s_id in seeds:
            s_name = seeds_config.get(s_id, f"Id{s_id}")
            pct = pool_attribs[s_id] / total_attrib if total_attrib > 0 else 0.0
            print(f"      * {s_name}: {pool_attribs[s_id]} candidates ({pct:.1%})")
            
        print("  - Representation Share (in final Top 10 recommendations):")
        for s_name, pct in final_shares.items():
            print(f"      * {s_name}: {pct:.1%}")
            
    print()

if __name__ == "__main__":
    main()
