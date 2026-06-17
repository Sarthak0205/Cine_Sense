import os
import sys
import numpy as np
import pandas as pd

# Ensure cinesense is importable
sys.path.insert(0, os.path.abspath("."))

from cinesense.utils.model_storage import load_model
from cinesense.services.recommendation import get_franchise, RecommendationService
from cinesense.retrieval.hybrid_c import hybrid_c_retrieval_scores, top_retrieval_indices

def main():
    model_dir = "cinesense/models/twostage_v1"
    model, catalog_df, _ = load_model(model_dir)
    service = RecommendationService(model, catalog_df)
    
    seeds = [1535, 1575, 9253]
    ratings = {1535: 10.0, 1575: 9.0, 9253: 8.0}
    
    dn_id, cg_id, sg_id = 1535, 1575, 9253
    dn_idx = service.recommender.item_id_to_index[dn_id]
    cg_idx = service.recommender.item_id_to_index[cg_id]
    sg_idx = service.recommender.item_id_to_index[sg_id]
    
    emb_dn = service.recommender.catalog_embeddings[dn_idx]
    emb_cg = service.recommender.catalog_embeddings[cg_idx]
    emb_sg = service.recommender.catalog_embeddings[sg_idx]
    
    train_indices = np.asarray([service.recommender.item_id_to_index[aid] for aid in seeds], dtype=np.int32)
    train_weights = np.array([1.0, 0.9, 0.8], dtype=np.float32)
    
    # 1. Stage-1 Retrieval to get Top 150
    ret_scores = hybrid_c_retrieval_scores(
        train_indices,
        service.recommender.catalog_embeddings,
        service.recommender.popularity_scores,
        service.recommender.semantic_weight,
        service.recommender.popularity_weight,
        service.recommender.seed_batch_size
    )
    
    train_items = set(seeds)
    ret_indices = top_retrieval_indices(
        ret_scores,
        train_items,
        service.recommender.anime_ids,
        150
    )
    
    # 2. Stage-2 Reranking scores
    batch_scores = service.recommender.catalog_embeddings @ service.recommender.catalog_embeddings[train_indices].T
    max_sim_array = batch_scores.max(axis=1)
    weighted_sim_sum = (batch_scores * train_weights.reshape(1, -1)).sum(axis=1)
    weighted_avg_array = weighted_sim_sum / np.sum(train_weights)
    
    hybrid_semantic_score = 0.7 * weighted_avg_array + 0.3 * max_sim_array
    
    rerank_scores = (
        service.recommender.semantic_weight * hybrid_semantic_score
        + service.recommender.popularity_weight * service.recommender.popularity_scores
    )
    
    # Process each candidate in the Top 150 pool
    candidates_data = []
    
    for i, idx in enumerate(ret_indices):
        aid = int(service.recommender.anime_ids[idx])
        title = service.catalog_meta[aid]["title"]
        
        # Retrieval score
        r_score = float(ret_scores[idx])
        
        # Rerank components
        sem_comp = float(service.recommender.semantic_weight * hybrid_semantic_score[idx])
        pop_comp = float(service.recommender.popularity_weight * service.recommender.popularity_scores[idx])
        rr_score = float(rerank_scores[idx])
        
        # Winning seed based on raw similarity
        emb_rec = service.recommender.catalog_embeddings[idx]
        sim_dn = float(np.dot(emb_rec, emb_dn))
        sim_cg = float(np.dot(emb_rec, emb_cg))
        sim_sg = float(np.dot(emb_rec, emb_sg))
        
        sims = [("Death Note", sim_dn), ("Code Geass", sim_cg), ("Steins;Gate", sim_sg)]
        winning_seed, _ = max(sims, key=lambda x: x[1])
        
        candidates_data.append({
            "id": aid,
            "title": title,
            "winning_seed": winning_seed,
            "ret_score": r_score,
            "sem_comp": sem_comp,
            "pop_comp": pop_comp,
            "rr_score": rr_score,
            "index": idx
        })
        
    # --- Table A: Average values by winning seed in the Top 150 pool ---
    print("\n=== TABLE A - AVERAGE VALUES BY WINNING SEED (TOP 150) ===")
    print("%-15s | %-12s | %-14s | %-15s" % ("Seed", "Avg Semantic", "Avg Popularity", "Avg Final Score"))
    print("-" * 65)
    for seed in ["Death Note", "Code Geass", "Steins;Gate"]:
        seed_cands = [c for c in candidates_data if c["winning_seed"] == seed]
        if seed_cands:
            avg_sem = np.mean([c["sem_comp"] for c in seed_cands])
            avg_pop = np.mean([c["pop_comp"] for c in seed_cands])
            avg_rr = np.mean([c["rr_score"] for c in seed_cands])
            print("%-15s | %12.4f | %14.4f | %15.4f" % (seed, avg_sem, avg_pop, avg_rr))
        else:
            print("%-15s | %-12s | %-14s | %-15s" % (seed, "N/A", "N/A", "N/A"))
            
    # --- Table B: Top 50 candidate counts before reranking ---
    # Top 50 sorted by retrieval score
    before_sorted = sorted(candidates_data, key=lambda c: -c["ret_score"])[:50]
    print("\n=== TABLE B - TOP 50 CANDIDATE COUNTS BEFORE RERANKING ===")
    print("%-15s | %-5s | %-10s" % ("Seed", "Count", "Percentage"))
    print("-" * 35)
    for seed in ["Death Note", "Code Geass", "Steins;Gate"]:
        cnt = sum(1 for c in before_sorted if c["winning_seed"] == seed)
        print("%-15s | %-5d | %8.1f%%" % (seed, cnt, (cnt / 50.0) * 100))
        
    # --- Table C: Top 50 counts after reranking ---
    # Top 50 sorted by rerank score
    after_sorted = sorted(candidates_data, key=lambda c: -c["rr_score"])[:50]
    print("\n=== TABLE C - TOP 50 CANDIDATE COUNTS AFTER RERANKING ===")
    print("%-15s | %-5s | %-10s" % ("Seed", "Count", "Percentage"))
    print("-" * 35)
    for seed in ["Death Note", "Code Geass", "Steins;Gate"]:
        cnt = sum(1 for c in after_sorted if c["winning_seed"] == seed)
        print("%-15s | %-5d | %8.1f%%" % (seed, cnt, (cnt / 50.0) * 100))
        
    # --- Table D: Score Delta ---
    print("\n=== TABLE D - SCORE DELTA ===")
    print("%-15s | %-18s | %-18s | %-10s" % ("Seed", "Avg Final Score", "Avg Retrieval Score", "Delta"))
    print("-" * 65)
    for seed in ["Death Note", "Code Geass", "Steins;Gate"]:
        seed_cands = [c for c in candidates_data if c["winning_seed"] == seed]
        if seed_cands:
            avg_rr = np.mean([c["rr_score"] for c in seed_cands])
            avg_ret = np.mean([c["ret_score"] for c in seed_cands])
            delta = avg_rr - avg_ret
            print("%-15s | %18.4f | %18.4f | %10.4f" % (seed, avg_rr, avg_ret, delta))
        else:
            print("%-15s | %-18s | %-18s | %-10s" % (seed, "N/A", "N/A", "N/A"))

if __name__ == "__main__":
    main()
