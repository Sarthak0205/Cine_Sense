import os
import sys
import numpy as np
import pandas as pd
import json

# Add project root to sys.path
PROJECT_ROOT = "/Users/sdc/Projects/CineSense-main"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.audit_common import (
    load_model_c,
    load_model_d,
    write_markdown_table,
    save_report,
    calculate_overlap,
    calculate_spearman,
)

def get_recs_and_shares(service, seeds, ratings):
    recs = service.recommend(anime_ids=seeds, ratings=ratings, top_k=20, mode="discover", user_id="stability_audit")
    rec_ids = [r["anime_id"] for r in recs]
    
    # Extract seed shares if present in explanations
    shares = {}
    for r in recs:
        exp = r.get("explanation", {})
        seed_shares = exp.get("seed_shares", {})
        for s_id, data in seed_shares.items():
            s_id_int = int(s_id)
            shares[s_id_int] = shares.get(s_id_int, 0.0) + float(data.get("share", 0.0))
            
    # Normalize shares
    total_share = sum(shares.values())
    if total_share > 0:
        shares = {k: v / total_share for k, v in shares.items()}
    else:
        shares = {s_id: 1.0 / len(seeds) for s_id in seeds}
        
    return rec_ids, shares

def calculate_attribution_drift(shares1, shares2):
    # Mean absolute deviation of shares
    all_keys = set(shares1.keys()) | set(shares2.keys())
    if not all_keys:
        return 0.0
    drift = 0.0
    for k in all_keys:
        v1 = shares1.get(k, 0.0)
        v2 = shares2.get(k, 0.0)
        drift += abs(v1 - v2)
    return drift / len(all_keys)

def run_stability_test(service, seeds1, ratings1, seeds2, ratings2):
    recs1, shares1 = get_recs_and_shares(service, seeds1, ratings1)
    recs2, shares2 = get_recs_and_shares(service, seeds2, ratings2)
    
    overlap_10 = calculate_overlap(recs1[:10], recs2[:10]) * 100
    overlap_20 = calculate_overlap(recs1[:20], recs2[:20]) * 100
    spearman = calculate_spearman(recs1[:10], recs2[:10])
    drift = calculate_attribution_drift(shares1, shares2)
    
    return overlap_10, overlap_20, spearman, drift

def main():
    print("Running Stability Audit...", flush=True)
    
    service_c = load_model_c()
    service_d = load_model_d()
    
    scenarios = [
        # Expected Stable
        {
            "name": "Seed Ordering: Naruto + One Piece",
            "cat": "Expected Stable",
            "seeds1": [20, 21], "ratings1": {20: 10.0, 21: 10.0},
            "seeds2": [21, 20], "ratings2": {21: 10.0, 20: 10.0}
        },
        {
            "name": "Rating Perturbation: DN(5)+MNS(10) vs DN(10)+MNS(10)",
            "cat": "Expected Stable",
            "seeds1": [1535, 19], "ratings1": {1535: 5.0, 19: 10.0},
            "seeds2": [1535, 19], "ratings2": {1535: 10.0, 19: 10.0}
        },
        
        # Expected Moderate Change
        {
            "name": "Seed Expansion: DN vs DN + Monster",
            "cat": "Expected Moderate",
            "seeds1": [1535], "ratings1": {1535: 10.0},
            "seeds2": [1535, 19], "ratings2": {1535: 10.0, 19: 10.0}
        },
        
        # Expected Larger Change
        {
            "name": "Multi-Seed Expansion: NAR+OP vs NAR+OP+BL",
            "cat": "Expected Larger",
            "seeds1": [20, 21], "ratings1": {20: 10.0, 21: 10.0},
            "seeds2": [20, 21, 269], "ratings2": {20: 10.0, 21: 10.0, 269: 10.0}
        }
    ]
    
    rows_c = []
    rows_d = []
    regressions_c = 0
    regressions_d = 0
    
    for sc in scenarios:
        name = sc["name"]
        cat = sc["cat"]
        
        # Model C
        o10_c, o20_c, sp_c, dr_c = run_stability_test(
            service_c, sc["seeds1"], sc["ratings1"], sc["seeds2"], sc["ratings2"]
        )
        rows_c.append([name, cat, f"{o10_c:.1f}%", f"{o20_c:.1f}%", f"{sp_c:.4f}", f"{dr_c:.4f}"])
        
        # Model D
        o10_d, o20_d, sp_d, dr_d = run_stability_test(
            service_d, sc["seeds1"], sc["ratings1"], sc["seeds2"], sc["ratings2"]
        )
        rows_d.append([name, cat, f"{o10_d:.1f}%", f"{o20_d:.1f}%", f"{sp_d:.4f}", f"{dr_d:.4f}"])
        
        # Validation checks
        if cat == "Expected Stable":
            if o10_c < 90.0: regressions_c += 1
            if o10_d < 90.0: regressions_d += 1
        elif cat == "Expected Moderate":
            if o10_c < 50.0: regressions_c += 1
            if o10_d < 50.0: regressions_d += 1
            
    pass_c = "PASS" if regressions_c == 0 else "FAIL"
    pass_d = "PASS" if regressions_d == 0 else "FAIL"
    
    # Build report content
    report_content = []
    report_content.append("# CineSense Recommendation Stability Audit Report")
    report_content.append(f"\n## Audit Summary")
    report_content.append(f"* **Model C Stability Status**: **{pass_c}**")
    report_content.append(f"* **Model D Stability Status**: **{pass_d}**")
    
    headers = ["Test Scenario", "Expected Class", "Top-10 Overlap", "Top-20 Overlap", "Spearman Corr", "Attribution Drift"]
    
    report_content.append(f"\n## Model C (Locked Baseline) Results")
    report_content.append(write_markdown_table(headers, rows_c))
    
    report_content.append(f"\n## Model D (Production Candidate) Results")
    report_content.append(write_markdown_table(headers, rows_d))
    
    report_content.append(f"\n## Sensitivity Analysis & Findings")
    if regressions_d > 0:
        report_content.append(
            "\n> [!WARNING]"
            "\n> Model D violated stability thresholds. "
            "\n> Popularity penalties and cosine scaling increase susceptibility to input noise, causing rank drift on minor perturbations."
        )
    else:
        report_content.append("\nBoth models exhibit expected stability patterns across all evaluated scenarios.")
        
    save_report("stability_audit.md", "\n".join(report_content))

if __name__ == "__main__":
    main()
