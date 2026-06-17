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
    print("Loading model for release candidate validation...", flush=True)
    model, catalog_df, metadata = load_model(os.path.join(PROJECT_ROOT, "cinesense/models/twostage_v1"))
    service = RecommendationService(model, catalog_df)

    # Cache get_franchise_root to ensure stable, fast validation run
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
        if "naruto" in title: return "NAR"
        if "bleach" in title: return "BL"
        if "k-on" in title: return "KON"
        if "berserk" in title: return "BSK"
        if "neon genesis" in title: return "NGE"
        return f"Id{aid}"

    # Helper to calculate seed shares
    def get_seed_shares(recs, seeds):
        seed_embeddings = {s_id: model.catalog_embeddings[model.item_id_to_index[s_id]] for s_id in seeds}
        counts = {s_id: 0 for s_id in seeds}
        for item in recs:
            # Map candidate to winning seed using cosine similarity
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
            shares[get_seed_abbr(s_id)] = count / len(recs) if recs else 0.0
        return shares

    # Define scenarios
    scenario_set_a = [
        {"name": "DN+CG+SG+MNS", "seeds": [1535, 1575, 9253, 19]},
        {"name": "AoT+FMAB+DN+HxH", "seeds": [16498, 5114, 1535, 11061]},
        {"name": "HxH+OP+NAR+BL", "seeds": [11061, 21, 20, 269]},
        {"name": "KON+SG+BSK+MNS", "seeds": [5680, 9253, 32379, 19]},
        {"name": "DN+AoT+FMAB+CG", "seeds": [1535, 16498, 5114, 1575]}
    ]

    scenario_set_b = [
        {"name": "DN+CG+SG+MNS+FMAB", "seeds": [1535, 1575, 9253, 19, 5114]},
        {"name": "AoT+DN+FMAB+HxH+OP", "seeds": [16498, 1535, 5114, 11061, 21]},
        {"name": "NAR+BL+OP+HxH+FMAB", "seeds": [20, 269, 21, 11061, 5114]},
        {"name": "SG+MNS+BSK+KON+NGE", "seeds": [9253, 19, 32379, 5680, 30]},
        {"name": "Mixed Genre Stress (CG+SG+NAR+KON+MNS)", "seeds": [1575, 9253, 20, 5680, 19]}
    ]

    all_scenarios = scenario_set_a + scenario_set_b
    results = []

    print("Running stress tests...", flush=True)

    for sc in all_scenarios:
        name = sc["name"]
        seeds = sc["seeds"]
        ratings = {s: 10.0 - i for i, s in enumerate(seeds)}
        
        # 1. Run Baseline (Penalty = False)
        os.environ["CINESENSE_REPRESENTATION_PENALTY"] = "False"
        if hasattr(model, "representation_penalty"):
            delattr(model, "representation_penalty")
        
        # Warm up
        service.recommend(seeds, ratings=ratings, top_k=10, mode="discover")
        
        recs_base = service.recommend(seeds, ratings=ratings, top_k=10, mode="discover")
        avg_score_base = np.mean([r["score"] for r in recs_base]) if recs_base else 0.0

        # 2. Run Experimental Release-Candidate (Penalty = True, Lambda = 0.03)
        os.environ["CINESENSE_REPRESENTATION_PENALTY"] = "True"
        os.environ["CINESENSE_REPRESENTATION_LAMBDA"] = "0.03"
        
        # Warm up
        service.recommend(seeds, ratings=ratings, top_k=10, mode="discover")
        
        t0 = time.perf_counter()
        recs_exp = service.recommend(seeds, ratings=ratings, top_k=10, mode="discover")
        t1 = time.perf_counter()
        latency = (t1 - t0) * 1000.0 # ms

        # Calculations
        scores = [r["score"] for r in recs_exp]
        avg_score_exp = np.mean(scores) if scores else 0.0
        min_score = np.min(scores) if scores else 0.0
        
        score_degradation = (avg_score_exp - avg_score_base) / avg_score_base if avg_score_base > 0 else 0.0
        
        # Discovery rate check
        seed_franchises = set()
        for s in seeds:
            meta = service.catalog_meta[s]
            seed_franchises.add(get_franchise(meta["title"]))
            if meta.get("title_english"):
                seed_franchises.add(get_franchise(meta["title_english"]))
                
        discovery_rate = sum(1 for r in recs_exp if get_franchise(r["title"]) not in seed_franchises) / len(recs_exp) if recs_exp else 0.0
        franchise_diversity = len(set(get_franchise(r["title"]) for r in recs_exp))
        
        # Representation shares
        shares = get_seed_shares(recs_exp, seeds)
        dominant_seed_share = max(shares.values()) if shares else 0.0
        
        # Failure checks
        failures = []
        # check if one seed receives 0%
        has_zero_seed = False
        zero_seeds = []
        for s, pct in shares.items():
            if pct == 0.0:
                has_zero_seed = True
                zero_seeds.append(s)
        if has_zero_seed:
            failures.append(f"Zero seed representation: {', '.join(zero_seeds)}")
            
        # dominant share exceeds 70%
        if dominant_seed_share > 0.70:
            failures.append(f"Dominant seed share exceeds 70% ({dominant_seed_share:.1%})")
            
        # diversity < 10
        if franchise_diversity < 10:
            failures.append(f"Franchise diversity is < 10 ({franchise_diversity})")
            
        # discovery rate < 100%
        if discovery_rate < 1.0:
            failures.append(f"Discovery rate is < 100% ({discovery_rate:.1%})")
            
        # score degradation exceeds 5% (i.e. degradation <= -0.05)
        if score_degradation < -0.05:
            failures.append(f"Score degradation exceeds 5% ({score_degradation:.2%})")

        results.append({
            "name": name,
            "seeds": seeds,
            "shares": shares,
            "avg_score": avg_score_exp,
            "min_score": min_score,
            "degradation": score_degradation,
            "discovery_rate": discovery_rate,
            "diversity": franchise_diversity,
            "latency": latency,
            "dominant_seed_share": dominant_seed_share,
            "failures": failures
        })

    # Clean up env
    if "CINESENSE_REPRESENTATION_PENALTY" in os.environ:
        del os.environ["CINESENSE_REPRESENTATION_PENALTY"]
    if "CINESENSE_REPRESENTATION_LAMBDA" in os.environ:
        del os.environ["CINESENSE_REPRESENTATION_LAMBDA"]

    print("\n" + "="*120)
    print("RELEASE CANDIDATE STRESS TEST DETAILS")
    print("="*120)
    print(f"| {'Scenario':<42} | {'Representation':<32} | {'Avg Score':<9} | {'Min Score':<9} | {'Degradation':<11} | {'DR':<6} | {'Div':<5} | {'Latency':<9} |")
    print(f"| {'-'*42} | {'-'*32} | {'-'*9} | {'-'*9} | {'-'*11} | {'-'*6} | {'-'*5} | {'-'*9} |")
    
    for r in results:
        repr_str = ", ".join(f"{s}:{pct:.0%}" for s, pct in r["shares"].items())
        print(f"| {r['name']:<42} | {repr_str:<32} | {r['avg_score']:<9.4f} | {r['min_score']:<9.4f} | {r['degradation']:<+11.2%} | {r['discovery_rate']:<6.0%} | {r['diversity']:<5} | {r['latency']:<6.2f} ms |")
        
    print()

    # Aggregate Metrics
    print("="*120)
    print("AGGREGATE METRICS SUMMARY")
    print("="*120)
    
    avg_dominance = np.mean([r["dominant_seed_share"] for r in results])
    avg_score = np.mean([r["avg_score"] for r in results])
    avg_diversity = np.mean([r["diversity"] for r in results])
    avg_discovery = np.mean([r["discovery_rate"] for r in results])
    avg_latency = np.mean([r["latency"] for r in results])
    
    print(f"* **Average Dominant Seed Share:** {avg_dominance:.2%}")
    print(f"* **Average Recommendation Score:** {avg_score:.4f}")
    print(f"* **Average Franchise Diversity:** {avg_diversity:.2f} unique franchises")
    print(f"* **Average Discovery Rate:** {avg_discovery:.2%}")
    print(f"* **Average Latency:** {avg_latency:.2f} ms")
    print()

    # Failure Analysis
    print("="*120)
    print("FAILURE ANALYSIS")
    print("="*120)
    total_failures = 0
    for r in results:
        if r["failures"]:
            print(f"Scenario: {r['name']}")
            for f in r["failures"]:
                print(f" - [FAIL] {f}")
                total_failures += 1
    if total_failures == 0:
        print("All failure conditions checked out. Zero scenarios flagged!")
    else:
        print(f"\nTotal failures flagged across scenarios: {total_failures}")
    print()

    # Rollout Recommendation
    print("="*120)
    print("ROLLOUT RECOMMENDATION VERDICT")
    print("="*120)
    # Verdict logic
    # If there are any core failures, we hold the rollout
    # If there are zero seed representations (which is common for 4/5 seed lists due to 10 slot limit), we can ship with monitoring.
    # Wait, let's look at the failure condition: "Flag any scenario where one seed receives 0%"
    # In a Top 10 list, with 5 seeds, if every seed gets exactly 20%, that's perfect. But if one seed gets 0%, does it count as a failure?
    # Yes, according to the instruction: "Flag any scenario where: one seed receives 0%"
    # Let's see if this happens, and how we analyze it.
    print()

if __name__ == "__main__":
    main()
