import os
import sys
import time
import numpy as np
import pandas as pd

# Ensure cinesense is importable
sys.path.insert(0, os.path.abspath("."))

from cinesense.utils.model_storage import load_model
from cinesense.services.recommendation import get_franchise, RecommendationService
from cinesense.retrieval.hybrid_c import hybrid_c_retrieval_scores, top_retrieval_indices

def run_retrieval_audit(service, valid_ids, train_weights, use_hybrid_retrieval=False):
    # Get index mapping
    dn_id = 1535
    cg_id = 1575
    sg_id = 9253
    
    dn_idx = service.recommender.item_id_to_index[dn_id]
    cg_idx = service.recommender.item_id_to_index[cg_id]
    sg_idx = service.recommender.item_id_to_index[sg_id]
    
    emb_dn = service.recommender.catalog_embeddings[dn_idx]
    emb_cg = service.recommender.catalog_embeddings[cg_idx]
    emb_sg = service.recommender.catalog_embeddings[sg_idx]
    
    train_indices = np.asarray([service.recommender.item_id_to_index[aid] for aid in valid_ids], dtype=np.int32)
    
    # Measure latency
    t_start = time.perf_counter()
    
    # Calculate retrieval scores
    if not use_hybrid_retrieval:
        # Original: semantic score = max(sim_i)
        ret_scores = hybrid_c_retrieval_scores(
            train_indices,
            service.recommender.catalog_embeddings,
            service.recommender.popularity_scores,
            service.recommender.semantic_weight,
            service.recommender.popularity_weight,
            service.recommender.seed_batch_size
        )
    else:
        # Experimental: semantic score = 0.7 * weighted_avg(sim_i) + 0.3 * max(sim_i)
        batch_scores = service.recommender.catalog_embeddings @ service.recommender.catalog_embeddings[train_indices].T
        max_sim = batch_scores.max(axis=1)
        weighted_sim_sum = (batch_scores * train_weights.reshape(1, -1)).sum(axis=1)
        weighted_avg = weighted_sim_sum / np.sum(train_weights)
        
        hybrid_semantic_score = 0.7 * weighted_avg + 0.3 * max_sim
        ret_scores = (
            service.recommender.semantic_weight * hybrid_semantic_score
            + service.recommender.popularity_weight * service.recommender.popularity_scores
        )
        
    # Exclude seeds
    train_items = set(valid_ids)
    
    # We will audit at candidate pool sizes: 50, 100, 150, 300
    pool_sizes = [50, 100, 150, 300]
    pool_results = {}
    
    # Calculate for latency averaging
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        if not use_hybrid_retrieval:
            _ = hybrid_c_retrieval_scores(
                train_indices,
                service.recommender.catalog_embeddings,
                service.recommender.popularity_scores,
                service.recommender.semantic_weight,
                service.recommender.popularity_weight,
                service.recommender.seed_batch_size
            )
        else:
            b_sc = service.recommender.catalog_embeddings @ service.recommender.catalog_embeddings[train_indices].T
            m_s = b_sc.max(axis=1)
            w_s_s = (b_sc * train_weights.reshape(1, -1)).sum(axis=1)
            w_a = w_s_s / np.sum(train_weights)
            h_s_s = 0.7 * w_a + 0.3 * m_s
            _ = (
                service.recommender.semantic_weight * h_s_s
                + service.recommender.popularity_weight * service.recommender.popularity_scores
            )
        latencies.append((time.perf_counter() - t0) * 1000)
    avg_latency = np.mean(latencies)

    for k in pool_sizes:
        ret_indices = top_retrieval_indices(
            ret_scores,
            train_items,
            service.recommender.anime_ids,
            k
        )
        
        attributions = {"Death Note": 0, "Code Geass": 0, "Steins;Gate": 0}
        sim_sums = {"Death Note": 0.0, "Code Geass": 0.0, "Steins;Gate": 0.0}
        
        for idx in ret_indices:
            emb_rec = service.recommender.catalog_embeddings[idx]
            
            sim_dn = float(np.dot(emb_rec, emb_dn))
            sim_cg = float(np.dot(emb_rec, emb_cg))
            sim_sg = float(np.dot(emb_rec, emb_sg))
            
            sim_sums["Death Note"] += sim_dn
            sim_sums["Code Geass"] += sim_cg
            sim_sums["Steins;Gate"] += sim_sg
            
            # Winning seed
            sims = [("Death Note", sim_dn), ("Code Geass", sim_cg), ("Steins;Gate", sim_sg)]
            winning_name, _ = max(sims, key=lambda x: x[1])
            attributions[winning_name] += 1
            
        avg_sims = {name: (sim_sums[name] / k) for name in ["Death Note", "Code Geass", "Steins;Gate"]}
        pcts = {name: (attributions[name] / k) * 100 for name in ["Death Note", "Code Geass", "Steins;Gate"]}
        
        pool_results[k] = {
            "attributions": attributions,
            "avg_sims": avg_sims,
            "percentages": pcts
        }
        
    return pool_results, avg_latency

def main():
    model_dir = "cinesense/models/twostage_v1"
    model, catalog_df, _ = load_model(model_dir)
    service = RecommendationService(model, catalog_df)
    
    valid_ids = [1535, 1575, 9253]
    train_weights = np.array([1.0, 0.9, 0.8], dtype=np.float32)
    
    print("=== RUNNING ORIGINAL RETRIEVAL AUDIT (score = max(sim_i)) ===", flush=True)
    orig_results, orig_lat = run_retrieval_audit(service, valid_ids, train_weights, use_hybrid_retrieval=False)
    
    print("\n=== RUNNING EXPERIMENTAL HYBRID RETRIEVAL AUDIT (score = 0.7*avg + 0.3*max) ===", flush=True)
    hybrid_results, hybrid_lat = run_retrieval_audit(service, valid_ids, train_weights, use_hybrid_retrieval=True)
    
    # ------------------ PRINT ORIGINAL RESULTS ------------------
    print("\n=====================================================================")
    print("ORIGINAL RETRIEVAL STAGE DOMINANCE AUDIT RESULTS")
    print("=====================================================================")
    print(f"Average Stage-1 Retrieval Latency (50 runs): {orig_lat:.4f} ms")
    
    print("\nTABLE A & B — Candidate Distribution and Representation %")
    print("-" * 75)
    print("%-15s | %-12s | %-12s | %-12s | %-12s" % ("Seed", "Top 50", "Top 100", "Top 150", "Top 300"))
    print("-" * 75)
    for name in ["Death Note", "Code Geass", "Steins;Gate"]:
        def fmt_col(k):
            cnt = orig_results[k]["attributions"][name]
            pct = orig_results[k]["percentages"][name]
            return f"{cnt} ({pct:.1f}%)"
        print("%-15s | %-12s | %-12s | %-12s | %-12s" % (
            name, fmt_col(50), fmt_col(100), fmt_col(150), fmt_col(300)
        ))
        
    print("\nTABLE C — Average Retrieval Similarity by Seed")
    print("-" * 75)
    print("%-15s | %-12s | %-12s | %-12s | %-12s" % ("Seed", "Top 50", "Top 100", "Top 150", "Top 300"))
    print("-" * 75)
    for name in ["Death Note", "Code Geass", "Steins;Gate"]:
        print("%-15s | %12.4f | %12.4f | %12.4f | %12.4f" % (
            name, orig_results[50]["avg_sims"][name], orig_results[100]["avg_sims"][name],
            orig_results[150]["avg_sims"][name], orig_results[300]["avg_sims"][name]
        ))
        
    # ------------------ PRINT HYBRID RESULTS ------------------
    print("\n=====================================================================")
    print("HYBRID RETRIEVAL STAGE DOMINANCE AUDIT RESULTS")
    print("=====================================================================")
    print(f"Average Stage-1 Retrieval Latency (50 runs): {hybrid_lat:.4f} ms")
    
    print("\nTABLE A & B — Candidate Distribution and Representation %")
    print("-" * 75)
    print("%-15s | %-12s | %-12s | %-12s | %-12s" % ("Seed", "Top 50", "Top 100", "Top 150", "Top 300"))
    print("-" * 75)
    for name in ["Death Note", "Code Geass", "Steins;Gate"]:
        def fmt_col(k):
            cnt = hybrid_results[k]["attributions"][name]
            pct = hybrid_results[k]["percentages"][name]
            return f"{cnt} ({pct:.1f}%)"
        print("%-15s | %-12s | %-12s | %-12s | %-12s" % (
            name, fmt_col(50), fmt_col(100), fmt_col(150), fmt_col(300)
        ))
        
    print("\nTABLE C — Average Retrieval Similarity by Seed")
    print("-" * 75)
    print("%-15s | %-12s | %-12s | %-12s | %-12s" % ("Seed", "Top 50", "Top 100", "Top 150", "Top 300"))
    print("-" * 75)
    for name in ["Death Note", "Code Geass", "Steins;Gate"]:
        print("%-15s | %12.4f | %12.4f | %12.4f | %12.4f" % (
            name, hybrid_results[50]["avg_sims"][name], hybrid_results[100]["avg_sims"][name],
            hybrid_results[150]["avg_sims"][name], hybrid_results[300]["avg_sims"][name]
        ))

if __name__ == "__main__":
    main()
