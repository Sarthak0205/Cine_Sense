import os
import sys
import numpy as np
import pandas as pd

# Set PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cinesense.utils.model_storage import load_model
from cinesense.services.recommendation import RecommendationService, get_franchise
from cinesense.ranking.weighted_b import weighted_max_similarity_to_train_items

def main():
    # Load model
    model, catalog_df, metadata = load_model(os.path.join(PROJECT_ROOT, "cinesense/models/twostage_v1"))
    service = RecommendationService(model, catalog_df)

    # Seeds and ratings (Death Note, Code Geass, Steins;Gate)
    seeds = [1535, 1575, 9253]
    ratings = {1535: 10.0, 1575: 9.0, 9253: 8.0}

    # Run experiment via the recommender and service by setting environment variables
    # This is a safe way to simulate both configs through the production recommendation pipeline
    def run_recommender(use_penalty, lmb):
        os.environ["CINESENSE_REPRESENTATION_PENALTY"] = "True" if use_penalty else "False"
        os.environ["CINESENSE_REPRESENTATION_LAMBDA"] = str(lmb)
        
        # Clear any existing penalty flags on the model to use env vars
        if hasattr(model, "representation_penalty"):
            delattr(model, "representation_penalty")
            
        # Get recommendations using production API discover mode
        recs = service.recommend(
            anime_ids=seeds,
            ratings=ratings,
            top_k=10,
            mode="discover"
        )
        return recs

    # Generate Baseline (Lambda = 0.00)
    baseline_recs = run_recommender(False, 0.00)
    
    # Generate Experimental (Lambda = 0.03)
    experimental_recs = run_recommender(True, 0.03)

    # Let's run a sweep over all lambda values to generate Tables A, B, C, D
    lambdas = [0.00, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
    sweep_results = {}
    for lmb in lambdas:
        sweep_results[lmb] = run_recommender(lmb > 0.0, lmb)

    # Clean up environment variables
    if "CINESENSE_REPRESENTATION_PENALTY" in os.environ:
        del os.environ["CINESENSE_REPRESENTATION_PENALTY"]
    if "CINESENSE_REPRESENTATION_LAMBDA" in os.environ:
        del os.environ["CINESENSE_REPRESENTATION_LAMBDA"]

    print("\n" + "="*80)
    print("CINESENSE REPRESENTATION PENALTY SIMULATION AUDIT")
    print("="*80 + "\n")

    # Table A: Seed representation in Top 10
    print("### Table A — Seed Attribution Distribution")
    print(f"| {'Lambda':<6} | {'Death Note (10)':<16} | {'Code Geass (9)':<15} | {'Steins;Gate (8)':<16} |")
    print(f"| {'-'*6} | {'-'*16} | {'-'*15} | {'-'*16} |")
    for lmb in lambdas:
        recs = sweep_results[lmb]
        counts = {1535: 0, 1575: 0, 9253: 0}
        for item in recs:
            ws = item["explanation"].get("matched_seed", {}).get("anime_id")
            if ws in counts:
                counts[ws] += 1
        print(f"| {lmb:<6.2f} | {counts[1535]:<16} | {counts[1575]:<15} | {counts[9253]:<16} |")
    print()

    # Table B: Top 10 recommendation comparison
    print("### Table B — Top 10 Recommendation Comparison")
    print(f"| {'Rank':<4} | {'Baseline (Lambda = 0.00)':<55} | {'Experimental (Lambda = 0.03)':<55} |")
    print(f"| {'-'*4} | {'-'*55} | {'-'*55} |")
    for i in range(10):
        base_title = ""
        exp_title = ""
        if i < len(baseline_recs):
            base_item = baseline_recs[i]
            base_title = base_item["title_english"] if base_item["title_english"] else base_item["title"]
            base_ws = "DN" if base_item["explanation"].get("matched_seed", {}).get("anime_id") == 1535 else ("CG" if base_item["explanation"].get("matched_seed", {}).get("anime_id") == 1575 else "SG")
            base_title = f"{base_title} ({base_ws})"
        if i < len(experimental_recs):
            exp_item = experimental_recs[i]
            exp_title = exp_item["title_english"] if exp_item["title_english"] else exp_item["title"]
            exp_ws = "DN" if exp_item["explanation"].get("matched_seed", {}).get("anime_id") == 1535 else ("CG" if exp_item["explanation"].get("matched_seed", {}).get("anime_id") == 1575 else "SG")
            exp_title = f"{exp_title} ({exp_ws})"
        print(f"| {i+1:<4} | {base_title[:55]:<55} | {exp_title[:55]:<55} |")
    print()

    # Table C: Average score delta
    print("### Table C — Average Score Delta")
    print(f"| {'Lambda':<6} | {'Avg Score':<12} | {'Score Delta':<13} | {'Degradation %':<13} |")
    print(f"| {'-'*6} | {'-'*12} | {'-'*13} | {'-'*13} |")
    baseline_avg = np.mean([item["score"] for item in sweep_results[0.0]])
    for lmb in lambdas:
        recs = sweep_results[lmb]
        avg_score = np.mean([item["score"] for item in recs]) if recs else 0.0
        delta = avg_score - baseline_avg
        deg_pct = delta / baseline_avg
        print(f"| {lmb:<6.2f} | {avg_score:<12.4f} | {delta:<13.4f} | {deg_pct:<13.2%} |")
    print()

    # Table D: Discovery rate and franchise diversity
    print("### Table D — Discovery Rate and Franchise Diversity")
    print(f"| {'Lambda':<6} | {'Discovery Rate':<14} | {'Unique Franchises':<17} | {'Min Score':<10} |")
    print(f"| {'-'*6} | {'-'*14} | {'-'*17} | {'-'*10} |")
    for lmb in lambdas:
        recs = sweep_results[lmb]
        # Discovery rate is percentage of items that do not belong to seeds (all pass Discover filters so this is 100%)
        # Franchise diversity: count of unique franchises
        franchises = set()
        for item in recs:
            franchises.add(get_franchise(item["title"]))
        min_score = np.min([item["score"] for item in recs]) if recs else 0.0
        print(f"| {lmb:<6.2f} | {100.0:<14.1f}% | {len(franchises):<17} | {min_score:<10.4f} |")
    print()

if __name__ == "__main__":
    main()
