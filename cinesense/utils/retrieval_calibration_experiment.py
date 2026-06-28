import os
import sys
import json
import time
import math
import numpy as np
import pandas as pd

# Set PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cinesense.utils.model_storage import load_model
from cinesense.services.recommendation import RecommendationService, get_franchise
from cinesense.retrieval.hybrid_c import top_retrieval_indices
from cinesense.ranking.weighted_b import weighted_max_similarity_to_train_items, rerank_candidates

# Custom retrieval scores calculation supporting calibration modes
def compute_calibrated_retrieval_scores(
    train_indices: np.ndarray,
    catalog_embeddings: np.ndarray,
    popularity_scores: np.ndarray,
    semantic_weight: float,
    popularity_weight: float,
    mode: str,
) -> np.ndarray:
    num_items = len(catalog_embeddings)
    max_scores = np.full(num_items, -np.inf, dtype=np.float32)

    for idx in train_indices:
        emb_s = catalog_embeddings[idx]
        sims = catalog_embeddings @ emb_s
        
        if mode == "baseline":
            calibrated_sims = sims
        elif mode == "zscore":
            # Compute top 100 neighbors of seed (excluding self)
            sims_no_self = np.delete(sims, idx)
            sims_no_self_sorted = np.sort(sims_no_self)[::-1]
            top100 = sims_no_self_sorted[:100]
            mean_val = np.mean(top100)
            std_val = np.std(top100)
            if std_val < 1e-6:
                std_val = 1e-6
            calibrated_sims = (sims - mean_val) / std_val
        elif mode == "relative":
            # Compute top 50 neighbors of seed (excluding self)
            sims_no_self = np.delete(sims, idx)
            sims_no_self_sorted = np.sort(sims_no_self)[::-1]
            top50 = sims_no_self_sorted[:50]
            mean_val = np.mean(top50)
            if mean_val < 1e-6:
                mean_val = 1e-6
            calibrated_sims = sims / mean_val
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        max_scores = np.maximum(max_scores, calibrated_sims)

    return semantic_weight * max_scores + popularity_weight * popularity_scores

# Custom recommend method for isolated evaluation
def run_custom_recommend(
    service: RecommendationService,
    anime_ids: list[int],
    ratings: dict[int, float],
    calibration_mode: str,
    top_k: int = 10,
) -> dict:
    model = service.recommender
    valid_ids, validated_ratings = service.validate_inputs(anime_ids, ratings, top_k)
    if not valid_ids:
        return {"recs": [], "audit": {}}

    # 1. Retrieval
    retrieval_k = max(300, top_k * 10)
    train_indices = np.asarray([model.item_id_to_index[aid] for aid in valid_ids], dtype=np.int32)
    train_items = set(valid_ids)

    retrieval_scores = compute_calibrated_retrieval_scores(
        train_indices,
        model.catalog_embeddings,
        model.popularity_scores,
        model.semantic_weight,
        model.popularity_weight,
        mode=calibration_mode,
    )
    retrieved_indices_raw = top_retrieval_indices(
        retrieval_scores,
        train_items,
        model.anime_ids,
        retrieval_k,
    )

    # Seed franchise exclusion
    seed_franchises = set()
    for aid in valid_ids:
        meta = service.catalog_meta.get(aid)
        if meta:
            seed_franchises.add(get_franchise(meta["title"]))
            if meta.get("title_english"):
                seed_franchises.add(get_franchise(meta["title_english"]))

    retrieved_indices_prepared = []
    excluded_count = 0
    for r_idx in retrieved_indices_raw:
        anime_id = int(model.anime_ids[r_idx])
        meta = service.catalog_meta[anime_id]
        title = meta["title"]
        eng_title = meta.get("title_english") or ""
        cand_f = get_franchise(title)
        cand_f_eng = get_franchise(eng_title) if eng_title else ""

        if cand_f in seed_franchises or (cand_f_eng and cand_f_eng in seed_franchises):
            excluded_count += 1
            continue

        retrieved_indices_prepared.append(r_idx)
        if len(retrieved_indices_prepared) == 150:
            break

    retrieved_indices = np.asarray(retrieved_indices_prepared, dtype=np.int32)

    # 2. Ranking
    train_weights = np.asarray([
        model._rating_weight(int(validated_ratings[aid])) if aid in validated_ratings else 1.0
        for aid in valid_ids
    ], dtype=np.float32)

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

    # Greedy Selection with Penalty
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

    # 3. Post-processing
    scores = {
        rec_id: float(rerank_scores[model.item_id_to_index[rec_id]])
        for rec_id in recommendations
    }
    enriched = service.enrich_recommendations(
        recommendations,
        scores,
        valid_ids,
        weighted_semantic_scores=weighted_semantic_scores,
    )

    # Downstream Discover filters
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

    recs = filtered_enriched[:top_k]

    # Attribution & Analysis helpers
    raw_retrieved_ids = [int(model.anime_ids[idx]) for idx in retrieved_indices_raw]
    ranked_ids = [int(model.anime_ids[idx]) for idx in retrieved_indices_prepared]
    final_ids = [r["anime_id"] for r in recs]

    return {
        "recs": recs,
        "raw_retrieved_ids": raw_retrieved_ids,
        "ranked_ids": ranked_ids,
        "final_ids": final_ids,
    }

# Compute standard recommendation evaluation metrics
def compute_metrics(recs, good_titles, acceptable_titles):
    relevance_grades = []
    good_set = {t.lower().strip() for t in good_titles}
    acceptable_set = {t.lower().strip() for t in acceptable_titles}
    
    for item in recs:
        title = item.get("title", "").lower().strip()
        title_eng = (item.get("title_english") or "").lower().strip()
        
        is_good = (title in good_set) or (title_eng in good_set)
        is_acceptable = (title in acceptable_set) or (title_eng in acceptable_set)
        
        if is_good:
            relevance_grades.append(2)
        elif is_acceptable:
            relevance_grades.append(1)
        else:
            relevance_grades.append(0)
            
    while len(relevance_grades) < 10:
        relevance_grades.append(0)
        
    p5 = sum(1 for r in relevance_grades[:5] if r > 0) / 5.0
    p10 = sum(1 for r in relevance_grades[:10] if r > 0) / 10.0
    
    total_relevant = len(good_set) + len(acceptable_set)
    r10 = sum(1 for r in relevance_grades[:10] if r > 0) / float(total_relevant) if total_relevant > 0 else 0.0
    
    dcg10 = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(relevance_grades[:10]))
        
    ideal_grades = [2] * len(good_set) + [1] * len(acceptable_set)
    ideal_grades.sort(reverse=True)
    ideal_grades = ideal_grades[:10]
    while len(ideal_grades) < 10:
        ideal_grades.append(0)
        
    idcg10 = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal_grades[:10]))
    ndcg10 = dcg10 / idcg10 if idcg10 > 0 else 0.0
    
    mrr = 0.0
    for i, r in enumerate(relevance_grades[:10]):
        if r > 0:
            mrr = 1.0 / (i + 1)
            break
            
    return p5, p10, r10, ndcg10, mrr

# Attribution Share Calculation Helper
def get_attribution_shares(item_ids, active_seeds, model):
    if not item_ids:
        return {s_id: 0.0 for s_id in active_seeds}
        
    seed_embeddings = {s_id: model.catalog_embeddings[model.item_id_to_index[s_id]] for s_id in active_seeds}
    counts = {s_id: 0 for s_id in active_seeds}
    
    for item_id in item_ids:
        idx = model.item_id_to_index.get(item_id)
        if idx is None:
            continue
        emb_c = model.catalog_embeddings[idx]
        best_seed = None
        max_sim = -float('inf')
        for s_id, emb_s in seed_embeddings.items():
            sim = float(np.dot(emb_c, emb_s))
            if sim > max_sim:
                max_sim = sim
                best_seed = s_id
        if best_seed in counts:
            counts[best_seed] += 1
            
    total = sum(counts.values())
    if total == 0:
        return {s_id: 0.0 for s_id in active_seeds}
    return {s_id: counts[s_id] / float(total) for s_id in active_seeds}

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

    # Force representation penalty flags for reproducibility
    os.environ["CINESENSE_REPRESENTATION_PENALTY"] = "True"
    os.environ["CINESENSE_REPRESENTATION_LAMBDA"] = "0.03"

    modes = ["baseline", "zscore", "relative"]
    
    # ----------------------------------------------------
    # Part 1: Gold Standard Evaluation (Single Seed Quality)
    # ----------------------------------------------------
    print("\n" + "="*80)
    print("PART 1: GOLD STANDARD EVALUATION (SINGLE SEED QUALITY)")
    print("="*80, flush=True)
    
    gs_results = {m: [] for m in modes}
    for entry in gold_dataset:
        seed_name = entry["seed"]
        seed_id = entry["anime_id"]
        good_recs = entry["good_recommendations"]
        acc_recs = entry["acceptable_recommendations"]
        
        for m in modes:
            res = run_custom_recommend(service, [seed_id], {seed_id: 10.0}, m, top_k=10)
            p5, p10, r10, ndcg10, mrr = compute_metrics(res["recs"], good_recs, acc_recs)
            gs_results[m].append({
                "seed": seed_name,
                "P5": p5,
                "P10": p10,
                "R10": r10,
                "NDCG10": ndcg10,
                "MRR": mrr
            })

    print(f"| {'Method':<25} | {'Mean P@5':<10} | {'Mean P@10':<10} | {'Mean R@10':<10} | {'Mean NDCG@10':<12} | {'Mean MRR':<10} |")
    print(f"| {'-'*25} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*12} | {'-'*10} |")
    for m in modes:
        df_m = pd.DataFrame(gs_results[m])
        print(f"| {m:<25} | {df_m['P5'].mean():<10.2%} | {df_m['P10'].mean():<10.2%} | {df_m['R10'].mean():<10.2%} | {df_m['NDCG10'].mean():<12.4f} | {df_m['MRR'].mean():<10.4f} |")

    # ----------------------------------------------------
    # Part 2: Critical Stress Scenarios
    # ----------------------------------------------------
    print("\n" + "="*80)
    print("PART 2: CRITICAL STRESS SCENARIOS EVALUATION")
    print("="*80, flush=True)

    stress_scenarios = [
        {"name": "DN + CG", "seeds": [1535, 1575], "ratings": {1535: 10.0, 1575: 9.0}},
        {"name": "DN + CG + SG", "seeds": [1535, 1575, 9253], "ratings": {1535: 10.0, 1575: 9.0, 9253: 8.0}},
        {"name": "FMAB + DN", "seeds": [5114, 1535], "ratings": {5114: 10.0, 1535: 9.0}},
        {"name": "FMAB + SG", "seeds": [5114, 9253], "ratings": {5114: 10.0, 9253: 9.0}},
        {"name": "FMAB + DN + SG", "seeds": [5114, 1535, 9253], "ratings": {5114: 10.0, 1535: 9.0, 9253: 8.0}},
        {"name": "FMAB + DN + AoT + HxH", "seeds": [5114, 1535, 16498, 11061], "ratings": {5114: 10.0, 1535: 9.0, 16498: 8.0, 11061: 7.0}},
        {"name": "HxH + OP", "seeds": [11061, 21], "ratings": {11061: 10.0, 21: 9.0}},
        {"name": "AoT + DN + FMAB", "seeds": [16498, 1535, 5114], "ratings": {16498: 10.0, 1535: 9.0, 5114: 8.0}}
    ]

    seed_names_map = {
        1535: "DN",
        1575: "CG",
        9253: "SG",
        16498: "AoT",
        5114: "FMAB",
        11061: "HxH",
        21: "OP"
    }

    # Store candidate tracking for specific analyses
    # Format: target_id -> calibration_mode -> scenario -> {retrieved_count, ranked_count, final_count}
    target_ids = {5114: "FMAB", 21: "OP", 11061: "HxH", 9253: "SG"}
    target_tracking = {target_name: {m: {} for m in modes} for target_name in target_ids.values()}

    for scenario in stress_scenarios:
        sc_name = scenario["name"]
        seeds = scenario["seeds"]
        ratings = scenario["ratings"]
        
        print(f"\nScenario: {sc_name}")
        print("-" * 50)
        
        for m in modes:
            res = run_custom_recommend(service, seeds, ratings, m, top_k=10)
            
            # Compute attribution shares
            ret_share = get_attribution_shares(res["raw_retrieved_ids"], seeds, model)
            rnk_share = get_attribution_shares(res["ranked_ids"], seeds, model)
            fin_share = get_attribution_shares(res["final_ids"], seeds, model)
            
            print(f"  Method: {m:<10}")
            print(f"    Seed      | Retrieved Share | Ranked Share | Final Share |")
            print(f"    ----------|-----------------|--------------|-------------|")
            for s_id in seeds:
                s_name = seed_names_map.get(s_id, f"Id{s_id}")
                print(f"    {s_name:<9} | {ret_share[s_id]:<15.1%} | {rnk_share[s_id]:<12.1%} | {fin_share[s_id]:<11.1%} |")
            
            # Track counts for target analysis
            for t_id, t_name in target_ids.items():
                if t_id in seeds:
                    # Filter items in raw_retrieved_ids, ranked_ids, final_ids that are attributed to t_id
                    # We can define attribution by highest similarity to seed t_id specifically
                    # Let's count them
                    # Wait, is the count defined as "items attributed to seed X" or "X franchise items"?
                    # In standard terms, let's use winning seed attribution
                    
                    seed_embeddings = {s_id: model.catalog_embeddings[model.item_id_to_index[s_id]] for s_id in seeds}
                    def count_attributed(ids):
                        cnt = 0
                        for idx_item in ids:
                            idx = model.item_id_to_index.get(idx_item)
                            if idx is None:
                                continue
                            emb_c = model.catalog_embeddings[idx]
                            best_seed = None
                            max_sim = -float('inf')
                            for s_id, emb_s in seed_embeddings.items():
                                sim = float(np.dot(emb_c, emb_s))
                                if sim > max_sim:
                                    max_sim = sim
                                    best_seed = s_id
                            if best_seed == t_id:
                                cnt += 1
                        return cnt
                        
                    target_tracking[t_name][m][sc_name] = {
                        "retrieved": count_attributed(res["raw_retrieved_ids"]),
                        "ranked": count_attributed(res["ranked_ids"]),
                        "final": count_attributed(res["final_ids"])
                    }

    # ----------------------------------------------------
    # Part 3: Detailed Target Analysis Tables
    # ----------------------------------------------------
    print("\n" + "="*80)
    print("PART 3: SPECIAL TARGET ANALYSES")
    print("="*80, flush=True)

    for t_name in ["FMAB", "OP", "HxH", "SG"]:
        print(f"\n{t_name} Pipeline Candidate Flow Across Methods:")
        print("-" * 65)
        # Header
        print(f"| {'Scenario':<25} | {'Metric':<10} | {'Baseline':<9} | {'Z-Score':<9} | {'Relative':<9} |")
        print(f"|{'-'*25}|{'-'*10}|{'-'*9}|{'-'*9}|{'-'*9}|")
        
        # We find scenarios where this target is present
        target_scenarios = []
        for sc in stress_scenarios:
            # check if target seed is in scenario
            # IDs: FMAB=5114, OP=21, HxH=11061, SG=9253
            t_id = { "FMAB": 5114, "OP": 21, "HxH": 11061, "SG": 9253 }[t_name]
            if t_id in sc["seeds"]:
                target_scenarios.append(sc["name"])
                
        for sc_name in target_scenarios:
            for stage in ["retrieved", "ranked", "final"]:
                val_base = target_tracking[t_name]["baseline"][sc_name][stage]
                val_z = target_tracking[t_name]["zscore"][sc_name][stage]
                val_rel = target_tracking[t_name]["relative"][sc_name][stage]
                print(f"| {sc_name:<25} | {stage:<10} | {val_base:<9} | {val_z:<9} | {val_rel:<9} |")
            print(f"|{'-'*25}|{'-'*10}|{'-'*9}|{'-'*9}|{'-'*9}|")

if __name__ == "__main__":
    main()
