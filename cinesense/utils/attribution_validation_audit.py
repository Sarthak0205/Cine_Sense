import os
import sys
import numpy as np
import pandas as pd

# Set PYTHONPATH
PROJECT_ROOT = "/Users/sdc/Projects/CineSense-main"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cinesense.utils.model_storage import load_model
from cinesense.services.recommendation import RecommendationService, get_franchise

def main():
    print("Loading model for attribution validation audit...", flush=True)
    model, catalog_df, metadata = load_model(os.path.join(PROJECT_ROOT, "cinesense/models/twostage_v1"))
    service = RecommendationService(model, catalog_df)

    # Cache get_franchise_root to ensure O(1) candidate post-processing
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

    # Audited scenarios
    scenarios = [
        {"name": "Death Note + Code Geass", "seeds": [1535, 1575]},
        {"name": "Death Note + Steins;Gate", "seeds": [1535, 9253]},
        {"name": "Death Note + Code Geass + Steins;Gate", "seeds": [1535, 1575, 9253]},
        {"name": "Monster + Death Note", "seeds": [19, 1535]},
        {"name": "AoT + FMAB", "seeds": [16498, 5114]},
        {"name": "Hunter x Hunter + Code Geass + Steins;Gate", "seeds": [11061, 1575, 9253]}
    ]

    all_recommendation_audits = []

    # Run recommendation under default production behavior (which currently has representation_penalty = False by default)
    # Ensure representation penalty is disabled
    model.representation_penalty = False
    if "CINESENSE_REPRESENTATION_PENALTY" in os.environ:
        del os.environ["CINESENSE_REPRESENTATION_PENALTY"]

    print("Running scenarios for audit...", flush=True)

    scenario_audits = {}

    for sc in scenarios:
        name = sc["name"]
        seeds = sc["seeds"]
        ratings = {s_id: 10.0 - i for i, s_id in enumerate(seeds)}
        
        # Get recommendations
        recs = service.recommend(seeds, ratings=ratings, top_k=10, mode="discover")
        
        # Precompute seed embeddings mapping for similarity calculations
        seed_embeddings = {s_id: model.catalog_embeddings[model.item_id_to_index[s_id]] for s_id in seeds}
        
        sc_recs_audits = []
        for rank, item in enumerate(recs, 1):
            cand_id = item["anime_id"]
            title = item["title_english"] if item["title_english"] else item["title"]
            emb_c = model.catalog_embeddings[model.item_id_to_index[cand_id]]
            
            # Calculate similarity to all seeds in the scenario
            sims = []
            for s_id in seeds:
                sim = float(np.dot(emb_c, seed_embeddings[s_id]))
                sims.append((s_id, sim))
            
            # Sort similarities descending
            sims.sort(key=lambda x: -x[1])
            
            best_seed, best_sim = sims[0]
            # Since all audited scenarios have >= 2 seeds, second seed always exists
            second_seed, second_sim = sims[1]
            gap = best_sim - second_sim
            
            audit_item = {
                "anime_id": cand_id,
                "title": title,
                "winning_seed": best_seed,
                "second_seed": second_seed,
                "best_similarity": best_sim,
                "second_similarity": second_sim,
                "similarity_gap": gap,
                "rank": rank
            }
            sc_recs_audits.append(audit_item)
            all_recommendation_audits.append(audit_item)
            
        scenario_audits[name] = {
            "recs": sc_recs_audits,
            "seeds": seeds
        }

    print("\n" + "="*80)
    print("CINESENSE ATTRIBUTION VALIDATION AUDIT")
    print("="*80 + "\n")

    # Deliverable 2 — Gap Distribution Analysis
    print("### Deliverable 2 — Gap Distribution Analysis")
    gaps = [item["similarity_gap"] for item in all_recommendation_audits]
    total_count = len(gaps)
    
    ranges = [
        ("< 0.02", lambda g: g < 0.02),
        ("0.02–0.05", lambda g: 0.02 <= g < 0.05),
        ("0.05–0.10", lambda g: 0.05 <= g < 0.10),
        ("> 0.10", lambda g: g >= 0.10)
    ]
    
    print(f"| {'Gap Range':<10} | {'Count':<6} | {'Percentage':<10} |")
    print(f"| {'-'*10} | {'-'*6} | {'-'*10} |")
    for label, cond in ranges:
        cnt = sum(1 for g in gaps if cond(g))
        pct = cnt / total_count if total_count > 0 else 0.0
        print(f"| {label:<10} | {cnt:<6} | {pct:<10.1%} |")
    print()

    # Deliverable 3 — Scenario Audit
    print("### Deliverable 3 — Scenario Audit")
    
    for sc in scenarios:
        name = sc["name"]
        data = scenario_audits[name]
        recs_audit = data["recs"]
        seeds = data["seeds"]
        
        print(f"\n#### Scenario: {name}")
        
        # 1. Attribution Table
        print("\n##### Attribution Table")
        print(f"| {'Rank':<4} | {'Title':<45} | {'Winning Seed':<12} | {'Second Seed':<12} | {'Gap':<8} |")
        print(f"| {'-'*4} | {'-'*45} | {'-'*12} | {'-'*12} | {'-'*8} |")
        for item in recs_audit:
            print(f"| {item['rank']:<4} | {item['title'][:45]:<45} | {get_seed_abbr(item['winning_seed']):<12} | {get_seed_abbr(item['second_seed']):<12} | {item['similarity_gap']:<8.4f} |")
            
        # 2. Representation Table
        print("\n##### Representation Table (Current Hard Attribution)")
        print(f"| {'Seed':<15} | {'Count':<6} | {'Share':<8} |")
        print(f"| {'-'*15} | {'-'*6} | {'-'*8} |")
        hard_counts = {s_id: 0 for s_id in seeds}
        for item in recs_audit:
            hard_counts[item["winning_seed"]] += 1
        for s_id in seeds:
            share = hard_counts[s_id] / len(recs_audit)
            print(f"| {get_seed_abbr(s_id):<15} | {hard_counts[s_id]:<6} | {share:<8.1%} |")
            
        # 3. Soft Attribution Table (Inverse-gap weighting: proportional split for gap < 0.05)
        print("\n##### Soft Attribution Table (Proportional Split for Gap < 0.05)")
        print(f"| {'Seed':<15} | {'Soft Share':<10} |")
        print(f"| {'-'*15} | {'-'*10} |")
        soft_weights = {s_id: 0.0 for s_id in seeds}
        for item in recs_audit:
            w_win = item["winning_seed"]
            w_sec = item["second_seed"]
            gap = item["similarity_gap"]
            best_sim = item["best_similarity"]
            sec_sim = item["second_similarity"]
            
            if gap < 0.05:
                # Proportional split based on similarity scores
                total_sim = best_sim + sec_sim
                w1 = best_sim / total_sim if total_sim > 0 else 0.5
                w2 = sec_sim / total_sim if total_sim > 0 else 0.5
                soft_weights[w_win] += w1
                soft_weights[w_sec] += w2
            else:
                soft_weights[w_win] += 1.0
                
        for s_id in seeds:
            soft_share = soft_weights[s_id] / len(recs_audit)
            print(f"| {get_seed_abbr(s_id):<15} | {soft_share:<10.1%} |")
            
    print()

if __name__ == "__main__":
    main()
