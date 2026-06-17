import os
import sys
import numpy as np
import pandas as pd

# Ensure cinesense is importable
sys.path.insert(0, os.path.abspath("."))

from cinesense.utils.model_storage import load_model
from cinesense.services.recommendation import RecommendationService

def main():
    model_dir = "cinesense/models/twostage_v1"
    model, catalog_df, _ = load_model(model_dir)
    service = RecommendationService(model, catalog_df)
    
    seeds = [1535, 1575, 9253]
    ratings = {1535: 10.0, 1575: 9.0, 9253: 8.0}
    
    # Generate Top 50 recommendations
    recs = service.recommend(seeds, ratings, top_k=50, mode="discover")
    
    print(f"Total recommendations returned: {len(recs)}")
    
    dn_id = 1535
    cg_id = 1575
    sg_id = 9253
    
    # Get index mapping
    dn_idx = service.recommender.item_id_to_index[dn_id]
    cg_idx = service.recommender.item_id_to_index[cg_id]
    sg_idx = service.recommender.item_id_to_index[sg_id]
    
    emb_dn = service.recommender.catalog_embeddings[dn_idx]
    emb_cg = service.recommender.catalog_embeddings[cg_idx]
    emb_sg = service.recommender.catalog_embeddings[sg_idx]
    
    # Check if seeds overlap (Embedding Collapse)
    dn_cg_sim = float(np.dot(emb_dn, emb_cg))
    dn_sg_sim = float(np.dot(emb_dn, emb_sg))
    cg_sg_sim = float(np.dot(emb_cg, emb_sg))
    
    print("\n[Embedding Seed Similarities]")
    print(f"  - Death Note vs Code Geass: {dn_cg_sim:.4f}")
    print(f"  - Death Note vs Steins;Gate: {dn_sg_sim:.4f}")
    print(f"  - Code Geass vs Steins;Gate: {cg_sg_sim:.4f}")
    
    table_data = []
    near_tie_count = 0
    
    for rank_idx, rec in enumerate(recs):
        rid = rec["anime_id"]
        title = rec["title"]
        score = rec["score"]
        
        # Look up embedding and compute raw similarities
        rec_idx = service.recommender.item_id_to_index[rid]
        emb_rec = service.recommender.catalog_embeddings[rec_idx]
        
        sim_dn = float(np.dot(emb_rec, emb_dn))
        sim_cg = float(np.dot(emb_rec, emb_cg))
        sim_sg = float(np.dot(emb_rec, emb_sg))
        
        # Sort similarities to find best and second best
        sims = [(dn_id, "Death Note", sim_dn), (cg_id, "Code Geass", sim_cg), (sg_id, "Steins;Gate", sim_sg)]
        sims_sorted = sorted(sims, key=lambda x: -x[2])
        
        winning_id, winning_name, max_similarity = sims_sorted[0]
        _, _, second_best_similarity = sims_sorted[1]
        
        # Near tie check
        is_near_tie = (max_similarity - second_best_similarity) < 0.05
        if is_near_tie:
            near_tie_count += 1
            
        table_data.append({
            "rank": rank_idx + 1,
            "id": rid,
            "title": title,
            "sim_dn": sim_dn,
            "sim_cg": sim_cg,
            "sim_sg": sim_sg,
            "winning_seed": winning_name,
            "is_near_tie": is_near_tie,
            "score": score
        })
        
    # Print the Top 50 detailed report
    print("\n%-5s | %-6s | %-30s | %-10s | %-10s | %-10s | %-15s | %-6s | %-8s" % (
        "Rank", "ID", "Title", "Sim DN", "Sim CG", "Sim SG", "Winning Seed", "Score", "Near-Tie"
    ))
    print("-" * 120)
    for row in table_data:
        print("%-5d | %-6d | %-30s | %10.4f | %10.4f | %10.4f | %-15s | %6.4f | %-8s" % (
            row["rank"], row["id"], row["title"][:30], row["sim_dn"], row["sim_cg"], row["sim_sg"],
            row["winning_seed"], row["score"], "YES" if row["is_near_tie"] else "NO"
        ))
        
    # Table A - Seed Attribution Distribution
    attributions = [row["winning_seed"] for row in table_data]
    print("\n=== TABLE A - SEED ATTRIBUTION DISTRIBUTION ===")
    print("%-15s | %-5s | %-10s" % ("Seed", "Count", "Percentage"))
    print("-" * 35)
    for seed_name in ["Death Note", "Code Geass", "Steins;Gate"]:
        count = attributions.count(seed_name)
        pct = (count / len(recs)) * 100 if recs else 0.0
        print("%-15s | %-5d | %8.1f%%" % (seed_name, count, pct))
        
    # Table B - Representation in Top 10 / Top 20 / Top 50
    print("\n=== TABLE B - REPRESENTATION ===")
    print("%-15s | %-6s | %-6s | %-6s" % ("Seed", "Top 10", "Top 20", "Top 50"))
    print("-" * 45)
    for seed_name in ["Death Note", "Code Geass", "Steins;Gate"]:
        count_10 = sum(1 for row in table_data[:10] if row["winning_seed"] == seed_name)
        count_20 = sum(1 for row in table_data[:20] if row["winning_seed"] == seed_name)
        count_50 = sum(1 for row in table_data[:50] if row["winning_seed"] == seed_name)
        print("%-15s | %-6d | %-6d | %-6d" % (seed_name, count_10, count_20, count_50))
        
    # Table C - Near-Tie Analysis
    pct_near_tie = (near_tie_count / len(recs)) * 100 if recs else 0.0
    print("\n=== TABLE C - NEAR-TIE ANALYSIS ===")
    print(f"  - Count of near-ties: {near_tie_count}")
    print(f"  - Percentage of near-ties: {pct_near_tie:.1f}%")

if __name__ == "__main__":
    main()
