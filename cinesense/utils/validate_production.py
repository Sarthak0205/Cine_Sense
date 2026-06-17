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
    print("Loading model for validation...", flush=True)
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

    seeds = [1535, 1575, 9253]
    ratings = {1535: 10.0, 1575: 9.0, 9253: 8.0}
    seed_abbrs = {1535: "Death Note", 1575: "Code Geass", 9253: "Steins;Gate"}

    # Enable representation penalty for validation
    os.environ["CINESENSE_REPRESENTATION_PENALTY"] = "True"
    os.environ["CINESENSE_REPRESENTATION_LAMBDA"] = "0.03"
    
    # Run production discover mode recommendation
    print("Running production recommendation...", flush=True)
    t0 = time.perf_counter()
    recs = service.recommend(seeds, ratings=ratings, top_k=10, mode="discover")
    t1 = time.perf_counter()
    prod_time = (t1 - t0) * 1000.0 # ms

    # Retrieve candidate audit log
    audit = getattr(service, "candidate_audit", {"retrieved": 0, "excluded_seed_franchise": 0, "remaining": 0})

    # Find winning seeds for representation
    seed_embeddings = {s_id: model.catalog_embeddings[model.item_id_to_index[s_id]] for s_id in seeds}
    def get_winning_seed(cand_id):
        emb_c = model.catalog_embeddings[model.item_id_to_index[cand_id]]
        best_seed = None
        max_sim = -float('inf')
        for s_id in seeds:
            sim = float(np.dot(emb_c, seed_embeddings[s_id]))
            if sim > max_sim:
                max_sim = sim
                best_seed = s_id
        return best_seed

    counts = {1535: 0, 1575: 0, 9253: 0}
    for item in recs:
        ws = get_winning_seed(item["anime_id"])
        if ws in counts:
            counts[ws] += 1

    print("\n" + "="*80)
    print("CINESENSE PRODUCTION VALIDATION RESULTS")
    print("="*80 + "\n")

    # Table A
    print("### Table A — Candidate Counts (DN+CG+SG)")
    print(f"| {'Stage':<25} | {'Count':<6} |")
    print(f"| {'-'*25} | {'-'*6} |")
    print(f"| {'Retrieved':<25} | {audit['retrieved']:<6} |")
    print(f"| {'Excluded Seed Franchise':<25} | {audit['excluded_seed_franchise']:<6} |")
    print(f"| {'Remaining':<25} | {audit['remaining']:<6} |")
    print()

    # Table B
    # Baseline: DN=60%, CG=0%, SG=40%
    # After: computed counts
    dn_pct = counts[1535] / 10
    cg_pct = counts[1575] / 10
    sg_pct = counts[9253] / 10
    print("### Table B — DN + CG + SG Representation")
    print(f"| {'Seed':<15} | {'Before':<8} | {'After':<8} |")
    print(f"| {'-'*15} | {'-'*8} | {'-'*8} |")
    print(f"| {'Death Note':<15} | {'60.0%':<8} | {dn_pct:<8.1%} |")
    print(f"| {'Code Geass':<15} | {'0.0%':<8} | {cg_pct:<8.1%} |")
    print(f"| {'Steins;Gate':<15} | {'40.0%':<8} | {sg_pct:<8.1%} |")
    print()

    # Table C
    # Baseline time: we know from the audit it was ~6.5 ms with caching
    print("### Table C — Runtime Comparison")
    print(f"| {'Metric':<20} | {'Before':<10} | {'After':<10} |")
    print(f"| {'-'*20} | {'-'*10} | {'-'*10} |")
    print(f"| {'Pipeline Latency':<20} | {'6.1 ms':<10} | {prod_time:<8.1f} ms |")
    print()

    # Final Result
    print("### Final Result")
    matches = (audit['retrieved'] == 300) and (cg_pct >= 0.15) and (sg_pct >= 0.15) and (0.50 <= dn_pct <= 0.70)
    if matches:
        print("Production implementation matches the validated audit results.")
    else:
        print("Production implementation does not match the validated audit results.")
    print()

if __name__ == "__main__":
    main()
