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

def main():
    print("Loading model for benchmark...", flush=True)
    model, catalog_df, metadata = load_model(os.path.join(PROJECT_ROOT, "cinesense/models/twostage_v1"))
    service = RecommendationService(model, catalog_df)
    
    # Helper to find ID by title search
    def find_id(query):
        matches = catalog_df[catalog_df["title"].str.contains(query, case=False, na=False) | 
                             catalog_df["title_english"].str.contains(query, case=False, na=False)]
        if len(matches) > 0:
            return int(matches.iloc[0]["anime_id"])
        return None

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
        if "naruto" in title: return "NAR"
        if "bleach" in title: return "BL"
        if "k-on" in title: return "KON"
        if "berserk" in title: return "BSK"
        if "kowabon" in title: return "KWB"
        return f"Id{aid}"

    # Determine stress test seed IDs
    naruto_id = find_id("naruto") or 20
    bleach_id = find_id("bleach") or 269
    one_piece_id = 21
    k_on_id = find_id("k-on") or 5680
    berserk_id = find_id("berserk") or 32379
    kowabon_id = find_id("kowabon") or 30948
    xanadu_id = find_id("xanadu") or 5920
    descendants_id = find_id("descendants of darkness") or 553

    # Define scenarios
    scenarios = [
        # Single Seed
        {"name": "1. Death Note", "seeds": [1535]},
        {"name": "2. Code Geass", "seeds": [1575]},
        {"name": "3. Steins;Gate", "seeds": [9253]},
        {"name": "4. Attack on Titan", "seeds": [16498]},
        {"name": "5. Fullmetal Alchemist: Brotherhood", "seeds": [5114]},
        {"name": "6. One Piece", "seeds": [21]},
        
        # Dual Seed
        {"name": "7. Death Note + Code Geass", "seeds": [1535, 1575]},
        {"name": "8. Attack on Titan + Code Geass", "seeds": [16498, 1575]},
        {"name": "9. Monster + Death Note", "seeds": [19, 1535]},
        {"name": "10. Hunter x Hunter + One Piece", "seeds": [11061, 21]},
        
        # Triple Seed
        {"name": "11. Death Note + Code Geass + Steins;Gate", "seeds": [1535, 1575, 9253]},
        {"name": "12. Death Note + Monster + Steins;Gate", "seeds": [1535, 19, 9253]},
        {"name": "13. AoT + Death Note + FMAB", "seeds": [16498, 1535, 5114]},
        {"name": "14. Hunter x Hunter + Code Geass + Steins;Gate", "seeds": [11061, 1575, 9253]},
        
        # Stress Tests
        {"name": "15. Three related shounen (NAR+BL+OP)", "seeds": [naruto_id, bleach_id, one_piece_id]},
        {"name": "16. Three unrelated (SG+KON+BSK)", "seeds": [9253, k_on_id, berserk_id]},
        {"name": "17. Three niche (KWB+XND+DES)", "seeds": [kowabon_id, xanadu_id, descendants_id]},
        {"name": "18. Three mainstream (DN+AoT+FMAB)", "seeds": [1535, 16498, 5114]},
    ]

    results = []

    for sc in scenarios:
        name = sc["name"]
        seeds = sc["seeds"]
        
        # Check if all seeds in index
        valid = True
        for s in seeds:
            if s not in model.item_id_to_index:
                print(f"Skipping seed {s} for scenario '{name}' (not in index)")
                valid = False
                break
        if not valid:
            continue
            
        ratings = {s: 10.0 - i for i, s in enumerate(seeds)} # DN 10.0, CG 9.0, SG 8.0, etc.

        # Run Baseline (Penalty = False)
        os.environ["CINESENSE_REPRESENTATION_PENALTY"] = "False"
        t0 = time.perf_counter()
        recs_base = service.recommend(seeds, ratings=ratings, top_k=10, mode="discover")
        t1 = time.perf_counter()
        time_base = (t1 - t0) * 1000.0 # ms

        # Run Penalty (Penalty = True)
        os.environ["CINESENSE_REPRESENTATION_PENALTY"] = "True"
        os.environ["CINESENSE_REPRESENTATION_LAMBDA"] = "0.03"
        t0 = time.perf_counter()
        recs_penalty = service.recommend(seeds, ratings=ratings, top_k=10, mode="discover")
        t2 = time.perf_counter()
        time_penalty = (t2 - t0) * 1000.0 # ms

        # Calculate metrics for Baseline
        base_scores = [r["score"] for r in recs_base]
        avg_score_base = np.mean(base_scores) if base_scores else 0.0
        
        seed_franchises = set()
        for s in seeds:
            meta = service.catalog_meta[s]
            seed_franchises.add(get_franchise(meta["title"]))
            if meta.get("title_english"):
                seed_franchises.add(get_franchise(meta["title_english"]))
                
        disc_rate_base = sum(1 for r in recs_base if get_franchise(r["title"]) not in seed_franchises) / len(recs_base) if recs_base else 0.0
        uniq_f_base = len(set(get_franchise(r["title"]) for r in recs_base))
        seq_cont_base = sum(1 for r in recs_base if service.is_sequel_title(r["title"]) or (r["title_english"] and service.is_sequel_title(r["title_english"])))

        # Baseline seed representation
        counts_base = {}
        for r in recs_base:
            ws = r["explanation"].get("matched_seed", {}).get("anime_id")
            if ws:
                counts_base[ws] = counts_base.get(ws, 0) + 1
        repr_base = ", ".join(f"{get_seed_abbr(ws)}:{cnt}" for ws, cnt in counts_base.items())

        # Calculate metrics for Penalty
        penalty_scores = [r["score"] for r in recs_penalty]
        avg_score_penalty = np.mean(penalty_scores) if penalty_scores else 0.0
        disc_rate_penalty = sum(1 for r in recs_penalty if get_franchise(r["title"]) not in seed_franchises) / len(recs_penalty) if recs_penalty else 0.0
        uniq_f_penalty = len(set(get_franchise(r["title"]) for r in recs_penalty))
        seq_cont_penalty = sum(1 for r in recs_penalty if service.is_sequel_title(r["title"]) or (r["title_english"] and service.is_sequel_title(r["title_english"])))

        # Penalty seed representation
        counts_penalty = {}
        for r in recs_penalty:
            ws = r["explanation"].get("matched_seed", {}).get("anime_id")
            if ws:
                counts_penalty[ws] = counts_penalty.get(ws, 0) + 1
        repr_penalty = ", ".join(f"{get_seed_abbr(ws)}:{cnt}" for ws, cnt in counts_penalty.items())

        results.append({
            "name": name,
            "seeds": seeds,
            "base": {
                "avg_score": avg_score_base,
                "disc_rate": disc_rate_base,
                "uniq_f": uniq_f_base,
                "seq_cont": seq_cont_base,
                "time": time_base,
                "repr": repr_base,
                "recs": recs_base
            },
            "penalty": {
                "avg_score": avg_score_penalty,
                "disc_rate": disc_rate_penalty,
                "uniq_f": uniq_f_penalty,
                "seq_cont": seq_cont_penalty,
                "time": time_penalty,
                "repr": repr_penalty,
                "recs": recs_penalty
            }
        })

    # Clean up env
    if "CINESENSE_REPRESENTATION_PENALTY" in os.environ:
        del os.environ["CINESENSE_REPRESENTATION_PENALTY"]
    if "CINESENSE_REPRESENTATION_LAMBDA" in os.environ:
        del os.environ["CINESENSE_REPRESENTATION_LAMBDA"]

    # Print Comparison Table
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON TABLE")
    print("="*80)
    print(f"| {'Scenario':<42} | {'DR Base/Pen':<12} | {'Div Base/Pen':<12} | {'Score Base/Pen':<15} | {'Time Base/Pen':<13} | {'Repr Base':<15} | {'Repr Pen':<15} |")
    print(f"| {'-'*42} | {'-'*12} | {'-'*12} | {'-'*15} | {'-'*13} | {'-'*15} | {'-'*15} |")
    
    avg_score_base_all = []
    avg_score_pen_all = []
    disc_rate_base_all = []
    disc_rate_pen_all = []
    uniq_f_base_all = []
    uniq_f_pen_all = []
    time_base_all = []
    time_pen_all = []

    for r in results:
        b = r["base"]
        p = r["penalty"]
        
        dr_str = f"{b['disc_rate']:.0%}/{p['disc_rate']:.0%}"
        div_str = f"{b['uniq_f']}/{p['uniq_f']}"
        score_str = f"{b['avg_score']:.4f}/{p['avg_score']:.4f}"
        time_str = f"{b['time']:.1f}/{p['time']:.1f}"
        
        print(f"| {r['name']:<42} | {dr_str:<12} | {div_str:<12} | {score_str:<15} | {time_str:<13} | {b['repr']:<15} | {p['repr']:<15} |")
        
        avg_score_base_all.append(b["avg_score"])
        avg_score_pen_all.append(p["avg_score"])
        disc_rate_base_all.append(b["disc_rate"])
        disc_rate_pen_all.append(p["disc_rate"])
        uniq_f_base_all.append(b["uniq_f"])
        uniq_f_pen_all.append(p["uniq_f"])
        time_base_all.append(b["time"])
        time_pen_all.append(p["time"])

    # Global Metrics Table
    print("\n" + "="*80)
    print("GLOBAL SUMMARY METRICS")
    print("="*80)
    print(f"| {'Metric':<30} | {'Baseline':<10} | {'Penalty':<10} | {'Delta':<10} |")
    print(f"| {'-'*30} | {'-'*10} | {'-'*10} | {'-'*10} |")
    
    mean_score_base = np.mean(avg_score_base_all)
    mean_score_pen = np.mean(avg_score_pen_all)
    mean_score_delta = mean_score_pen - mean_score_base
    mean_score_pct = mean_score_delta / mean_score_base
    print(f"| {'Avg Recommendation Score':<30} | {mean_score_base:<10.4f} | {mean_score_pen:<10.4f} | {mean_score_pct:<10.2%} |")

    mean_dr_base = np.mean(disc_rate_base_all)
    mean_dr_pen = np.mean(disc_rate_pen_all)
    mean_dr_delta = mean_dr_pen - mean_dr_base
    print(f"| {'Discovery Rate':<30} | {mean_dr_base:<10.1%} | {mean_dr_pen:<10.1%} | {mean_dr_delta:<10.1%} |")

    mean_div_base = np.mean(uniq_f_base_all)
    mean_div_pen = np.mean(uniq_f_pen_all)
    mean_div_delta = mean_div_pen - mean_div_base
    print(f"| {'Franchise Diversity':<30} | {mean_div_base:<10.2f} | {mean_div_pen:<10.2f} | {mean_div_delta:<+10.2f} |")

    mean_time_base = np.mean(time_base_all)
    mean_time_pen = np.mean(time_pen_all)
    mean_time_delta = mean_time_pen - mean_time_base
    mean_time_pct = mean_time_delta / mean_time_base
    print(f"| {'Runtime (ms)':<30} | {mean_time_base:<10.1f} | {mean_time_pen:<10.1f} | {mean_time_pct:<10.2%} |")

    # Evaluate promotion criteria
    print("\n" + "="*80)
    print("PROMOTION GATE EVALUATION")
    print("="*80)
    
    gate_passed = True
    reasons = []

    # 1. Code Geass representation >= 15% (for DN+CG+SG scenario: index 10)
    triple_scenario = next((r for r in results if "Death Note + Code Geass + Steins;Gate" in r["name"]), None)
    if triple_scenario:
        recs = triple_scenario["penalty"]["recs"]
        counts = {1535: 0, 1575: 0, 9253: 0}
        for item in recs:
            ws = item["explanation"].get("matched_seed", {}).get("anime_id")
            if ws in counts:
                counts[ws] += 1
        cg_share = counts[1575] / len(recs) if recs else 0.0
        sg_share = counts[9253] / len(recs) if recs else 0.0
        
        print(f"DN+CG+SG share: Code Geass = {cg_share:.1%}, Steins;Gate = {sg_share:.1%}")
        if cg_share < 0.15:
            gate_passed = False
            reasons.append(f"Code Geass representation in triple seed scenario is only {cg_share:.1%}, which is < 15%")
        if sg_share < 0.15:
            gate_passed = False
            reasons.append(f"Steins;Gate representation in triple seed scenario is only {sg_share:.1%}, which is < 15%")
    else:
        print("Warning: Triple seed scenario (DN+CG+SG) was not evaluated!")
        gate_passed = False
        reasons.append("Triple seed scenario (DN+CG+SG) was not evaluated")

    # 2. Average score degradation < 5%
    if mean_score_pct < -0.05:
        gate_passed = False
        reasons.append(f"Global average score degradation is {mean_score_pct:.2%}, which is >= 5% limit")

    # 3. Discovery rate unchanged
    if mean_dr_delta < -0.01:
        gate_passed = False
        reasons.append(f"Discovery rate dropped by {mean_dr_delta:.1%}")

    # 4. Franchise diversity unchanged
    if mean_div_delta < -0.1:
        gate_passed = False
        reasons.append(f"Franchise diversity decreased by {mean_div_delta:.2f}")

    # 5. Runtime increase < 10%
    if mean_time_pct > 0.10:
        gate_passed = False
        reasons.append(f"Runtime increased by {mean_time_pct:.2%}, exceeding the 10% threshold limit")

    # 6. Single seed output regressions
    single_seed_changed = False
    for r in results:
        if len(r["seeds"]) == 1:
            base_recs_ids = [item["anime_id"] for item in r["base"]["recs"]]
            pen_recs_ids = [item["anime_id"] for item in r["penalty"]["recs"]]
            if base_recs_ids != pen_recs_ids:
                single_seed_changed = True
                print(f"Regression: Single seed scenario '{r['name']}' outputs changed! Base: {base_recs_ids} | Penalty: {pen_recs_ids}")
    
    if single_seed_changed:
        gate_passed = False
        reasons.append("Single seed query output regression detected (results changed under representation_penalty=True)")

    if gate_passed:
        print("\nGATE STATUS: PASS - Ready for Production Promotion")
    else:
        print("\nGATE STATUS: FAIL - Keep representation_penalty=False")
        print("Detailed Reasons for Failure:")
        for r in reasons:
            print(f" - {r}")
            
if __name__ == "__main__":
    main()
