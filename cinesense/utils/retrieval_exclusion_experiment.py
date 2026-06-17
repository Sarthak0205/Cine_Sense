import os
import sys
import time
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
    print("Loading model for retrieval exclusion experiment...", flush=True)
    model, catalog_df, metadata = load_model(os.path.join(PROJECT_ROOT, "cinesense/models/twostage_v1"))
    service = RecommendationService(model, catalog_df)

    # Cache get_franchise_root to avoid high O(N) lookup complexity per candidate
    print("Optimizing service with caching...", flush=True)
    original_get_franchise_root = service.get_franchise_root
    franchise_root_cache = {}
    def cached_get_franchise_root(franchise_name):
        if franchise_name not in franchise_root_cache:
            franchise_root_cache[franchise_name] = original_get_franchise_root(franchise_name)
        return franchise_root_cache[franchise_name]
    service.get_franchise_root = cached_get_franchise_root

    # Helper to get seed abbreviations
    def get_seed_abbr(aid):
        title = service.catalog_meta.get(aid, {}).get("title", "").lower()
        if "death note" in title: return "DN"
        if "code geass" in title: return "CG"
        if "steins;gate" in title: return "SG"
        if "shingeki no kyojin" in title or "attack on titan" in title: return "AoT"
        if "fullmetal alchemist" in title: return "FMAB"
        if "one piece" in title: return "OP"
        if "monster" in title: return "MNS"
        if "hunter" in title: return "HxH"
        return f"Id{aid}"

    # Helper to find winning seed for candidate
    seed_embeddings = {}
    def get_winning_seed_id(cand_id, seeds_list):
        emb_c = model.catalog_embeddings[model.item_id_to_index[cand_id]]
        best_seed = None
        max_sim = -float('inf')
        for s_id in seeds_list:
            if s_id not in seed_embeddings:
                seed_embeddings[s_id] = model.catalog_embeddings[model.item_id_to_index[s_id]]
            sim = float(np.dot(emb_c, seed_embeddings[s_id]))
            if sim > max_sim:
                max_sim = sim
                best_seed = s_id
        return best_seed

    # Define pipelines
    def run_baseline_pipeline(seeds, ratings, top_k):
        # 1. Retrieval
        t_ret_0 = time.perf_counter()
        train_indices = np.asarray([model.item_id_to_index[aid] for aid in seeds], dtype=np.int32)
        train_weights = np.asarray([model._rating_weight(int(ratings[aid])) for aid in seeds], dtype=np.float32)
        retrieval_scores = hybrid_c_retrieval_scores(
            train_indices, model.catalog_embeddings, model.popularity_scores,
            model.semantic_weight, model.popularity_weight, model.seed_batch_size
        )
        retrieved_indices = top_retrieval_indices(retrieval_scores, set(seeds), model.anime_ids, 150)
        t_ret_1 = time.perf_counter()
        ret_time = (t_ret_1 - t_ret_0) * 1000.0 # ms
        
        # 2. Ranking
        t_rank_0 = time.perf_counter()
        weighted_semantic_scores = weighted_max_similarity_to_train_items(
            train_indices, train_weights, model.catalog_embeddings, model.seed_batch_size
        )
        rerank_scores = (
            model.semantic_weight * weighted_semantic_scores
            + model.popularity_weight * model.popularity_scores
        )
        ranked_anime_ids = rerank_candidates(
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
        t_rank_1 = time.perf_counter()
        rank_time = (t_rank_1 - t_rank_0) * 1000.0 # ms
        
        # 3. Discover Filters
        scores = {
            aid: float(rerank_scores[model.item_id_to_index[aid]])
            for aid in ranked_anime_ids
        }
        enriched = service.enrich_recommendations(ranked_anime_ids, scores, seeds, weighted_semantic_scores)
        
        seed_franchises = set()
        for aid in seeds:
            meta = service.catalog_meta.get(aid)
            if meta:
                seed_franchises.add(get_franchise(meta["title"]))
                if meta.get("title_english"):
                    seed_franchises.add(get_franchise(meta["title_english"]))
                    
        filtered_enriched = []
        seen_rec_franchises = set()
        for item in enriched:
            rec_id = item["anime_id"]
            rec_title = item["title"]
            rec_eng_title = item.get("title_english")
            
            # A. Exclude seed franchises
            rec_f_name = get_franchise(rec_title)
            rec_f_eng_name = get_franchise(rec_eng_title) if rec_eng_title else ""
            if rec_f_name in seed_franchises or (rec_f_eng_name and rec_f_eng_name in seed_franchises):
                continue
                
            # B. Sequel Filtering
            root_id = service.get_franchise_root(rec_f_name)
            is_sequel = False
            if root_id is not None and rec_id != root_id:
                is_sequel = True
            elif service.is_sequel_title(rec_title) or (rec_eng_title and service.is_sequel_title(rec_eng_title)):
                is_sequel = True
                
            if is_sequel:
                continue
                
            # C. Franchise Deduplication
            if rec_f_name in seen_rec_franchises or (rec_f_eng_name and rec_f_eng_name in seen_rec_franchises):
                continue
                
            filtered_enriched.append(item)
            seen_rec_franchises.add(rec_f_name)
            if rec_f_eng_name:
                seen_rec_franchises.add(rec_f_eng_name)
                
        return filtered_enriched[:top_k], ret_time, rank_time

    def run_experimental_pipeline(seeds, ratings, top_k):
        # 1. Retrieval
        t_ret_0 = time.perf_counter()
        train_indices = np.asarray([model.item_id_to_index[aid] for aid in seeds], dtype=np.int32)
        train_weights = np.asarray([model._rating_weight(int(ratings[aid])) for aid in seeds], dtype=np.float32)
        retrieval_scores = hybrid_c_retrieval_scores(
            train_indices, model.catalog_embeddings, model.popularity_scores,
            model.semantic_weight, model.popularity_weight, model.seed_batch_size
        )
        # Retrieve 300 candidates
        retrieved_indices_300 = top_retrieval_indices(retrieval_scores, set(seeds), model.anime_ids, 300)
        
        # 2. Discover-Aware Retrieval Preparation: Immediately exclude seed franchise items
        seed_franchises = set()
        for aid in seeds:
            meta = service.catalog_meta.get(aid)
            if meta:
                seed_franchises.add(get_franchise(meta["title"]))
                if meta.get("title_english"):
                    seed_franchises.add(get_franchise(meta["title_english"]))
                    
        retrieved_indices_prepared = []
        for idx in retrieved_indices_300:
            anime_id = int(model.anime_ids[idx])
            meta = service.catalog_meta[anime_id]
            title = meta["title"]
            eng_title = meta.get("title_english") or ""
            cand_f = get_franchise(title)
            cand_f_eng = get_franchise(eng_title) if eng_title else ""
            
            if cand_f in seed_franchises or (cand_f_eng and cand_f_eng in seed_franchises):
                continue
                
            retrieved_indices_prepared.append(idx)
            if len(retrieved_indices_prepared) == 150:
                break
                
        retrieved_indices_prepared = np.asarray(retrieved_indices_prepared, dtype=np.int32)
        t_ret_1 = time.perf_counter()
        ret_time = (t_ret_1 - t_ret_0) * 1000.0 # ms
        
        # 3. Ranking
        t_rank_0 = time.perf_counter()
        weighted_semantic_scores = weighted_max_similarity_to_train_items(
            train_indices, train_weights, model.catalog_embeddings, model.seed_batch_size
        )
        rerank_scores = (
            model.semantic_weight * weighted_semantic_scores
            + model.popularity_weight * model.popularity_scores
        )
        ranked_anime_ids = rerank_candidates(
            retrieved_indices_prepared,
            rerank_scores,
            retrieval_scores,
            model.anime_ids,
            150,
            representation_penalty=True,
            representation_lambda=0.03,
            train_indices=train_indices,
            catalog_embeddings=model.catalog_embeddings,
        )
        t_rank_1 = time.perf_counter()
        rank_time = (t_rank_1 - t_rank_0) * 1000.0 # ms
        
        # 4. Discover Filters
        scores = {
            aid: float(rerank_scores[model.item_id_to_index[aid]])
            for aid in ranked_anime_ids
        }
        enriched = service.enrich_recommendations(ranked_anime_ids, scores, seeds, weighted_semantic_scores)
        
        filtered_enriched = []
        seen_rec_franchises = set()
        for item in enriched:
            rec_id = item["anime_id"]
            rec_title = item["title"]
            rec_eng_title = item.get("title_english")
            
            # (Same franchise is already excluded during retrieval prep)
            
            # B. Sequel Filtering
            rec_f_name = get_franchise(rec_title)
            rec_f_eng_name = get_franchise(rec_eng_title) if rec_eng_title else ""
            root_id = service.get_franchise_root(rec_f_name)
            is_sequel = False
            if root_id is not None and rec_id != root_id:
                is_sequel = True
            elif service.is_sequel_title(rec_title) or (rec_eng_title and service.is_sequel_title(rec_eng_title)):
                is_sequel = True
                
            if is_sequel:
                continue
                
            # C. Franchise Deduplication
            if rec_f_name in seen_rec_franchises or (rec_f_eng_name and rec_f_eng_name in seen_rec_franchises):
                continue
                
            filtered_enriched.append(item)
            seen_rec_franchises.add(rec_f_name)
            if rec_f_eng_name:
                seen_rec_franchises.add(rec_f_eng_name)
                
        return filtered_enriched[:top_k], ret_time, rank_time

    # Scenarios config
    scenarios = [
        # Single Seed
        {"name": "DN", "seeds": [1535]},
        {"name": "CG", "seeds": [1575]},
        {"name": "SG", "seeds": [9253]},
        {"name": "AoT", "seeds": [16498]},
        {"name": "FMAB", "seeds": [5114]},
        
        # Dual Seed
        {"name": "DN+CG", "seeds": [1535, 1575]},
        {"name": "AoT+CG", "seeds": [16498, 1575]},
        {"name": "MNS+DN", "seeds": [19, 1535]},
        {"name": "HxH+OP", "seeds": [11061, 21]},
        
        # Triple Seed
        {"name": "DN+CG+SG", "seeds": [1535, 1575, 9253]},
        {"name": "DN+MNS+SG", "seeds": [1535, 19, 9253]},
        {"name": "AoT+DN+FMAB", "seeds": [16498, 1535, 5114]},
        {"name": "HxH+CG+SG", "seeds": [11061, 1575, 9253]}
    ]

    print("Running scenarios...", flush=True)
    sc_results = []
    
    for sc in scenarios:
        name = sc["name"]
        seeds = sc["seeds"]
        
        # Ratings logic (DN=10, CG=9, SG=8, etc.)
        ratings = {s_id: 10.0 - idx for idx, s_id in enumerate(seeds)}
        
        # Run Baseline (top_k=50 to allow representation measuring at 10, 20, 50)
        recs_base, base_ret_time, base_rank_time = run_baseline_pipeline(seeds, ratings, 50)
        
        # Run Experimental (top_k=50)
        recs_exp, exp_ret_time, exp_rank_time = run_experimental_pipeline(seeds, ratings, 50)
        
        sc_results.append({
            "name": name,
            "seeds": seeds,
            "ratings": ratings,
            "base": {
                "recs": recs_base,
                "ret_time": base_ret_time,
                "rank_time": base_rank_time
            },
            "exp": {
                "recs": recs_exp,
                "ret_time": exp_ret_time,
                "rank_time": exp_rank_time
            }
        })

    # Calculations for deliverables
    print("\n" + "="*80)
    print("CINESENSE RETRIEVAL-STAGE FRANCHISE EXCLUSION AUDIT RESULTS")
    print("="*80 + "\n")

    # Helper to calculate representation
    def calc_repr_string(recs, seeds_list, limit):
        sub = recs[:limit]
        if not sub:
            return "N/A"
        counts = {s_id: 0 for s_id in seeds_list}
        for item in sub:
            ws = get_winning_seed_id(item["anime_id"], seeds_list)
            if ws in counts:
                counts[ws] += 1
        return ", ".join(f"{get_seed_abbr(ws)}:{cnt/len(sub):.0%}" for ws, cnt in counts.items() if cnt > 0)

    # Deliverable 1: Representation tables
    print("### Deliverable 1 — Before/After Representation Tables")
    print("#### Top 10 Recommendations Pool")
    print(f"| {'Scenario':<15} | {'Baseline (Top 10)':<30} | {'Experimental (Top 10)':<30} |")
    print(f"| {'-'*15} | {'-'*30} | {'-'*30} |")
    for res in sc_results:
        b_str = calc_repr_string(res["base"]["recs"], res["seeds"], 10)
        e_str = calc_repr_string(res["exp"]["recs"], res["seeds"], 10)
        print(f"| {res['name']:<15} | {b_str:<30} | {e_str:<30} |")
    print()

    print("#### Top 20 Recommendations Pool")
    print(f"| {'Scenario':<15} | {'Baseline (Top 20)':<30} | {'Experimental (Top 20)':<30} |")
    print(f"| {'-'*15} | {'-'*30} | {'-'*30} |")
    for res in sc_results:
        b_str = calc_repr_string(res["base"]["recs"], res["seeds"], 20)
        e_str = calc_repr_string(res["exp"]["recs"], res["seeds"], 20)
        print(f"| {res['name']:<15} | {b_str:<30} | {e_str:<30} |")
    print()

    print("#### Top 50 Recommendations Pool")
    print(f"| {'Scenario':<15} | {'Baseline (Top 50)':<30} | {'Experimental (Top 50)':<30} |")
    print(f"| {'-'*15} | {'-'*30} | {'-'*30} |")
    for res in sc_results:
        b_str = calc_repr_string(res["base"]["recs"], res["seeds"], 50)
        e_str = calc_repr_string(res["exp"]["recs"], res["seeds"], 50)
        print(f"| {res['name']:<15} | {b_str:<30} | {e_str:<30} |")
    print()

    # Deliverable 2: Code Geass representation improvement
    print("### Deliverable 2 — Code Geass Representation Analysis (DN+CG+SG)")
    triple_res = next(r for r in sc_results if r["name"] == "DN+CG+SG")
    b_recs_10 = triple_res["base"]["recs"][:10]
    e_recs_10 = triple_res["exp"]["recs"][:10]
    
    # Calculate DN, CG, SG counts
    counts_b = {1535: 0, 1575: 0, 9253: 0}
    for item in b_recs_10:
        ws = get_winning_seed_id(item["anime_id"], triple_res["seeds"])
        if ws in counts_b: counts_b[ws] += 1
        
    counts_e = {1535: 0, 1575: 0, 9253: 0}
    for item in e_recs_10:
        ws = get_winning_seed_id(item["anime_id"], triple_res["seeds"])
        if ws in counts_e: counts_e[ws] += 1
        
    dn_b, cg_b, sg_b = counts_b[1535]/10, counts_b[1575]/10, counts_b[9253]/10
    dn_e, cg_e, sg_e = counts_e[1535]/10, counts_e[1575]/10, counts_e[9253]/10
    
    print(f"Baseline shares in DN+CG+SG: DN={dn_b:.0%}, CG={cg_b:.0%}, SG={sg_b:.0%}")
    print(f"Experimental shares in DN+CG+SG: DN={dn_e:.0%}, CG={cg_e:.0%}, SG={sg_e:.0%}")
    print()

    # Deliverable 3: Quality degradation analysis
    print("### Deliverable 3 — Quality Degradation Analysis (Top 10)")
    print(f"| {'Scenario':<15} | {'Base Avg/Min Score':<20} | {'Exp Avg/Min Score':<20} | {'Avg Score Delta':<15} | {'Div Base/Exp':<12} |")
    print(f"| {'-'*15} | {'-'*20} | {'-'*20} | {'-'*15} | {'-'*12} |")
    
    score_degs = []
    div_delta = []
    
    for res in sc_results:
        b_10 = res["base"]["recs"][:10]
        e_10 = res["exp"]["recs"][:10]
        
        b_scores = [item["score"] for item in b_10]
        e_scores = [item["score"] for item in e_10]
        
        b_avg = np.mean(b_scores) if b_scores else 0.0
        b_min = np.min(b_scores) if b_scores else 0.0
        e_avg = np.mean(e_scores) if e_scores else 0.0
        e_min = np.min(e_scores) if e_scores else 0.0
        
        avg_delta = (e_avg - b_avg) / b_avg if b_avg > 0 else 0.0
        score_degs.append(avg_delta)
        
        # Franchise Diversity
        f_b = len(set(get_franchise(item["title"]) for item in b_10))
        f_e = len(set(get_franchise(item["title"]) for item in e_10))
        div_delta.append(f_e - f_b)
        
        print(f"| {res['name']:<15} | {b_avg:.4f}/{b_min:.4f} | {e_avg:.4f}/{e_min:.4f} | {avg_delta:<+15.2%} | {f_b}/{f_e} |")
    print()

    # Deliverable 4: Runtime impact
    print("### Deliverable 4 — Runtime Impact (ms)")
    print(f"| {'Scenario':<15} | {'Base Ret/Rank':<15} | {'Exp Ret/Rank':<15} | {'Total Base/Exp':<15} | {'Delta':<10} |")
    print(f"| {'-'*15} | {'-'*15} | {'-'*15} | {'-'*15} | {'-'*10} |")
    
    total_base_time = 0.0
    total_exp_time = 0.0
    
    for res in sc_results:
        b = res["base"]
        e = res["exp"]
        
        t_base = b["ret_time"] + b["rank_time"]
        t_exp = e["ret_time"] + e["rank_time"]
        
        total_base_time += t_base
        total_exp_time += t_exp
        
        delta = (t_exp - t_base) / t_base if t_base > 0 else 0.0
        print(f"| {res['name']:<15} | {b['ret_time']:.1f}/{b['rank_time']:.1f} | {e['ret_time']:.1f}/{e['rank_time']:.1f} | {t_base:.1f}/{t_exp:.1f} | {delta:<+10.2%} |")
    
    mean_delta = (total_exp_time - total_base_time) / total_base_time
    print(f"\nMean Latency Impact: {mean_delta:+.2%} ({total_base_time/len(sc_results):.1f}ms -> {total_exp_time/len(sc_results):.1f}ms)")
    print()

    # Deliverable 5: Final Verdict
    print("### Final Verdict")
    success = True
    reasons = []
    
    # Check Code Geass target: CG >= 15%, SG >= 15%, DN = 50-70% in triple seed
    if cg_e < 0.15:
        success = False
        reasons.append(f"Code Geass representation is only {cg_e:.1%} (target >= 15%)")
    if sg_e < 0.15:
        success = False
        reasons.append(f"Steins;Gate representation is only {sg_e:.1%} (target >= 15%)")
    if not (0.50 <= dn_e <= 0.70):
        success = False
        reasons.append(f"Death Note representation is {dn_e:.1%} (target 50-70%)")
        
    # Check score degradation < 5% on average
    mean_score_deg = np.mean(score_degs)
    if mean_score_deg < -0.05:
        success = False
        reasons.append(f"Mean score degradation is {mean_score_deg:.2%} (limit < 5%)")
        
    # Check franchise diversity
    for res in sc_results:
        if len(res["seeds"]) > 1: # multi seed
            # Check unique franchises for Exp
            uniq = len(set(get_franchise(item["title"]) for item in res["exp"]["recs"][:10]))
            if uniq < 10 and len(res["exp"]["recs"][:10]) == 10:
                success = False
                reasons.append(f"Franchise diversity for scenario {res['name']} dropped to {uniq} (limit >= 10)")

    if success:
        print("VERDICT: PROCEED TO IMPLEMENT")
        print("\nQuantitative Justification:")
        print(f" - Code Geass representation improved to {cg_e:.0%} (baseline {cg_b:.0%}, target >= 15%)")
        print(f" - Steins;Gate representation is {sg_e:.0%} (baseline {sg_b:.0%}, target >= 15%)")
        print(f" - Death Note representation is balanced to {dn_e:.0%} (baseline {dn_b:.0%}, target 50-70%)")
        print(f" - Mean average recommendation score change is only {mean_score_deg:+.2%} (limit < 5%)")
        print(f" - Runtime impact is negligible (mean delta: {mean_delta:+.2%})")
    else:
        print("VERDICT: REJECT APPROACH")
        print("\nReasons for rejection:")
        for r in reasons:
            print(f" - {r}")
    print()

if __name__ == "__main__":
    main()
