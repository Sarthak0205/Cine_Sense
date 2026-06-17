import os
import sys
import re
import numpy as np
import pandas as pd

# Set PYTHONPATH
PROJECT_ROOT = "/Users/sdc/Projects/CineSense-main"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cinesense.utils.model_storage import load_model
from cinesense.services.recommendation import RecommendationService, get_franchise
from cinesense.retrieval.hybrid_c import hybrid_c_retrieval_scores, top_retrieval_indices
from cinesense.ranking.weighted_b import weighted_max_similarity_to_train_items

def main():
    # Load model
    print("Loading model for contamination audit...", flush=True)
    model, catalog_df, metadata = load_model(os.path.join(PROJECT_ROOT, "cinesense/models/twostage_v1"))
    service = RecommendationService(model, catalog_df)

    # Monkey-patch get_franchise_root with caching to avoid O(N_pool * N_catalog) runtime complexity
    print("Optimizing recommendation service with caching...", flush=True)
    original_get_franchise_root = service.get_franchise_root
    franchise_root_cache = {}
    def cached_get_franchise_root(franchise_name):
        if franchise_name not in franchise_root_cache:
            franchise_root_cache[franchise_name] = original_get_franchise_root(franchise_name)
        return franchise_root_cache[franchise_name]
    service.get_franchise_root = cached_get_franchise_root

    # Seeds list
    seeds = [
        {"id": 1575, "name": "Code Geass"},
        {"id": 1535, "name": "Death Note"},
        {"id": 9253, "name": "Steins;Gate"},
        {"id": 16498, "name": "Attack on Titan"},
        {"id": 5114, "name": "Fullmetal Alchemist: Brotherhood"},
        {"id": 11061, "name": "Hunter x Hunter"},
        {"id": 21, "name": "One Piece"},
    ]

    pool_sizes = [50, 100, 150, 300, 500]
    results = {}

    for seed in seeds:
        seed_id = seed["id"]
        seed_name = seed["name"]
        
        # Verify seed ID in index
        if seed_id not in model.item_id_to_index:
            continue
            
        seed_idx = model.item_id_to_index[seed_id]
        seed_emb = model.catalog_embeddings[seed_idx]
        seed_meta = service.catalog_meta[seed_id]
        seed_title = seed_meta["title"]
        seed_eng_title = seed_meta.get("title_english") or ""

        # Determine seed franchise names
        seed_franchises = {get_franchise(seed_title)}
        if seed_eng_title:
            seed_franchises.add(get_franchise(seed_eng_title))

        # Calculate cosine similarity
        similarities = model.catalog_embeddings @ seed_emb
        sorted_indices = np.argsort(-similarities)

        # Retrieve top 500 semantic neighbors (excluding the seed itself)
        neighbors = []
        for idx in sorted_indices:
            cand_id = int(model.anime_ids[idx])
            if cand_id == seed_id:
                continue
            neighbors.append((cand_id, float(similarities[idx]), idx))
            if len(neighbors) == 500:
                break

        results[seed_name] = {
            "neighbors": neighbors,
            "seed_id": seed_id,
            "seed_franchises": seed_franchises,
            "seed_emb": seed_emb
        }

    print("\n" + "="*80, flush=True)
    print("CINESENSE FRANCHISE NEIGHBORHOOD CONTAMINATION AUDIT", flush=True)
    print("="*80 + "\n", flush=True)

    # Step 3: Table A — Franchise Contamination Rates
    print("### Table A — Franchise Contamination Rates", flush=True)
    print(f"| {'Seed':<32} | {'Pool Size':<9} | {'Same Franchise':<14} | {'Different':<9} | {'Contamination %':<15} |", flush=True)
    print(f"| {'-'*32} | {'-'*9} | {'-'*14} | {'-'*9} | {'-'*15} |", flush=True)
    for seed_name, data in results.items():
        neighbors = data["neighbors"]
        seed_franchises = data["seed_franchises"]
        for p_size in pool_sizes:
            sub = neighbors[:p_size]
            same_count = 0
            for cand_id, _, _ in sub:
                meta = service.catalog_meta[cand_id]
                title = meta["title"]
                eng_title = meta.get("title_english") or ""
                cand_f = get_franchise(title)
                cand_f_eng = get_franchise(eng_title) if eng_title else ""
                if cand_f in seed_franchises or (cand_f_eng and cand_f_eng in seed_franchises):
                    same_count += 1
            diff_count = p_size - same_count
            pct = same_count / p_size
            print(f"| {seed_name:<32} | {p_size:<9} | {same_count:<14} | {diff_count:<9} | {pct:<15.1%} |", flush=True)
    print(flush=True)

    # Step 4: Table B — Discoverability Rates
    print("### Table B — Discoverability Rates", flush=True)
    print(f"| {'Seed':<32} | {'Pool Size':<9} | {'Discover Eligible':<17} | {'Discover %':<10} |", flush=True)
    print(f"| {'-'*32} | {'-'*9} | {'-'*17} | {'-'*10} |", flush=True)
    for seed_name, data in results.items():
        neighbors = data["neighbors"]
        seed_franchises = data["seed_franchises"]
        for p_size in pool_sizes:
            sub = neighbors[:p_size]
            discover_eligible = 0
            seen_rec_franchises = set()
            for cand_id, _, _ in sub:
                meta = service.catalog_meta[cand_id]
                title = meta["title"]
                eng_title = meta.get("title_english") or ""
                cand_f = get_franchise(title)
                cand_f_eng = get_franchise(eng_title) if eng_title else ""
                
                # Exclude seed franchise
                if cand_f in seed_franchises or (cand_f_eng and cand_f_eng in seed_franchises):
                    continue
                    
                # Sequel Filtering
                root_id = service.get_franchise_root(cand_f)
                is_sequel = False
                if root_id is not None and cand_id != root_id:
                    is_sequel = True
                elif service.is_sequel_title(title) or (eng_title and service.is_sequel_title(eng_title)):
                    is_sequel = True
                    
                if is_sequel:
                    continue
                    
                # Franchise Deduplication
                if cand_f in seen_rec_franchises or (cand_f_eng and cand_f_eng in seen_rec_franchises):
                    continue
                    
                discover_eligible += 1
                seen_rec_franchises.add(cand_f)
                if cand_f_eng:
                    seen_rec_franchises.add(cand_f_eng)
            print(f"| {seed_name:<32} | {p_size:<9} | {discover_eligible:<17} | {discover_eligible / p_size:<10.1%} |", flush=True)
    print(flush=True)

    # Step 5: Table C — Code Geass Rank Distribution
    print("### Table C — Code Geass Rank Distribution", flush=True)
    print(f"| {'Rank Range':<12} | {'Same Franchise Count':<20} |", flush=True)
    print(f"| {'-'*12} | {'-'*20} |", flush=True)
    cg_data = results["Code Geass"]
    cg_neighbors = cg_data["neighbors"]
    cg_franchises = cg_data["seed_franchises"]
    
    ranges = [
        ("1-25", 0, 25),
        ("26-50", 25, 50),
        ("51-100", 50, 100),
        ("101-150", 100, 150),
        ("151-300", 150, 300),
        ("301-500", 300, 500)
    ]
    for label, start, end in ranges:
        count = 0
        for cand_id, _, _ in cg_neighbors[start:end]:
            meta = service.catalog_meta[cand_id]
            title = meta["title"]
            eng_title = meta.get("title_english") or ""
            cand_f = get_franchise(title)
            cand_f_eng = get_franchise(eng_title) if eng_title else ""
            if cand_f in cg_franchises or (cand_f_eng and cand_f_eng in cg_franchises):
                count += 1
        print(f"| {label:<12} | {count:<20} |", flush=True)
    print(flush=True)

    # Step 6: Table D — First Discoverable Rank
    print("### Table D — First Discoverable Rank", flush=True)
    print(f"| {'Seed':<32} | {'First Discoverable Rank':<23} |", flush=True)
    print(f"| {'-'*32} | {'-'*23} |", flush=True)
    for seed_name, data in results.items():
        neighbors = data["neighbors"]
        seed_franchises = data["seed_franchises"]
        first_discoverable = "N/A"
        for rank, (cand_id, _, _) in enumerate(neighbors, 1):
            meta = service.catalog_meta[cand_id]
            title = meta["title"]
            eng_title = meta.get("title_english") or ""
            cand_f = get_franchise(title)
            cand_f_eng = get_franchise(eng_title) if eng_title else ""
            
            # Check Same Franchise
            if cand_f in seed_franchises or (cand_f_eng and cand_f_eng in seed_franchises):
                continue
                
            # Check Sequel
            root_id = service.get_franchise_root(cand_f)
            is_sequel = False
            if root_id is not None and cand_id != root_id:
                is_sequel = True
            elif service.is_sequel_title(title) or (eng_title and service.is_sequel_title(eng_title)):
                is_sequel = True
                
            if is_sequel:
                continue
                
            # First one found!
            first_discoverable = rank
            break
        print(f"| {seed_name:<32} | {first_discoverable:<23} |", flush=True)
    print(flush=True)

    # Step 7: Table E — Early Franchise Exclusion Simulation (Code Geass)
    print("### Table E — Early Franchise Exclusion Simulation", flush=True)
    
    cg_id = 1575
    cg_seed_emb = cg_data["seed_emb"]
    cg_seed_franchises = cg_data["seed_franchises"]
    
    # 1. Current Behavior (Retrieve 150 pool first, then apply filters)
    train_indices = np.asarray([model.item_id_to_index[cg_id]], dtype=np.int32)
    retrieval_scores = hybrid_c_retrieval_scores(
        train_indices,
        model.catalog_embeddings,
        model.popularity_scores,
        model.semantic_weight,
        model.popularity_weight,
        model.seed_batch_size,
    )
    retrieved_indices_curr = top_retrieval_indices(
        retrieval_scores,
        {cg_id},
        model.anime_ids,
        150,
    )
    
    # Rerank Current pool using Weighted B
    train_weights = np.asarray([1.0], dtype=np.float32)
    weighted_semantic_scores = weighted_max_similarity_to_train_items(
        train_indices,
        train_weights,
        model.catalog_embeddings,
        model.seed_batch_size,
    )
    rerank_scores = (
        model.semantic_weight * weighted_semantic_scores
        + model.popularity_weight * model.popularity_scores
    )
    reranked_indices_curr = sorted(
        retrieved_indices_curr,
        key=lambda idx: (
            -float(rerank_scores[idx]),
            -float(retrieval_scores[idx]),
            int(model.anime_ids[idx]),
        ),
    )
    
    # Apply filters to current pool
    recs_curr = []
    seen_franchises_curr = set()
    for idx in reranked_indices_curr:
        anime_id = int(model.anime_ids[idx])
        meta = service.catalog_meta[anime_id]
        title = meta["title"]
        eng_title = meta.get("title_english") or ""
        cand_f = get_franchise(title)
        cand_f_eng = get_franchise(eng_title) if eng_title else ""
        
        if cand_f in cg_seed_franchises or (cand_f_eng and cand_f_eng in cg_seed_franchises):
            continue
            
        root_id = service.get_franchise_root(cand_f)
        is_sequel = False
        if root_id is not None and anime_id != root_id:
            is_sequel = True
        elif service.is_sequel_title(title) or (eng_title and service.is_sequel_title(eng_title)):
            is_sequel = True
            
        if is_sequel:
            continue
            
        if cand_f in seen_franchises_curr or (cand_f_eng and cand_f_eng in seen_franchises_curr):
            continue
            
        sim = float(model.catalog_embeddings[idx] @ cg_seed_emb)
        score = float(rerank_scores[idx])
        recs_curr.append({
            "anime_id": anime_id,
            "title": title,
            "franchise": cand_f,
            "similarity": sim,
            "score": score
        })
        seen_franchises_curr.add(cand_f)
        if cand_f_eng:
            seen_franchises_curr.add(cand_f_eng)
            
        if len(recs_curr) == 10:
            break

    # 2. Experimental Behavior (Remove Same-Franchise items *first*, then retrieve 150)
    ranked_indices_all = np.argsort(-retrieval_scores, kind="mergesort")
    retrieved_indices_exp = []
    for idx in ranked_indices_all:
        anime_id = int(model.anime_ids[idx])
        if anime_id == cg_id:
            continue
        meta = service.catalog_meta[anime_id]
        title = meta["title"]
        eng_title = meta.get("title_english") or ""
        cand_f = get_franchise(title)
        cand_f_eng = get_franchise(eng_title) if eng_title else ""
        if cand_f in cg_seed_franchises or (cand_f_eng and cand_f_eng in cg_seed_franchises):
            continue
        retrieved_indices_exp.append(int(idx))
        if len(retrieved_indices_exp) == 150:
            break
            
    # Rerank Experimental pool using Weighted B
    reranked_indices_exp = sorted(
        retrieved_indices_exp,
        key=lambda idx: (
            -float(rerank_scores[idx]),
            -float(retrieval_scores[idx]),
            int(model.anime_ids[idx]),
        ),
    )
    
    # Apply filters to experimental pool
    recs_exp = []
    seen_franchises_exp = set()
    for idx in reranked_indices_exp:
        anime_id = int(model.anime_ids[idx])
        meta = service.catalog_meta[anime_id]
        title = meta["title"]
        eng_title = meta.get("title_english") or ""
        cand_f = get_franchise(title)
        cand_f_eng = get_franchise(eng_title) if eng_title else ""
        
        # Same Franchise is already filtered out during retrieval stage
        
        root_id = service.get_franchise_root(cand_f)
        is_sequel = False
        if root_id is not None and anime_id != root_id:
            is_sequel = True
        elif service.is_sequel_title(title) or (eng_title and service.is_sequel_title(eng_title)):
            is_sequel = True
            
        if is_sequel:
            continue
            
        if cand_f in seen_franchises_exp or (cand_f_eng and cand_f_eng in seen_franchises_exp):
            continue
            
        sim = float(model.catalog_embeddings[idx] @ cg_seed_emb)
        score = float(rerank_scores[idx])
        recs_exp.append({
            "anime_id": anime_id,
            "title": title,
            "franchise": cand_f,
            "similarity": sim,
            "score": score
        })
        seen_franchises_exp.add(cand_f)
        if cand_f_eng:
            seen_franchises_exp.add(cand_f_eng)
            
        if len(recs_exp) == 10:
            break

    # Format metrics comparison
    curr_len = len(recs_curr)
    exp_len = len(recs_exp)
    
    curr_uniq = len(seen_franchises_curr)
    exp_uniq = len(seen_franchises_exp)
    
    curr_avg_sim = np.mean([item["similarity"] for item in recs_curr]) if recs_curr else 0.0
    exp_avg_sim = np.mean([item["similarity"] for item in recs_exp]) if recs_exp else 0.0
    
    curr_avg_score = np.mean([item["score"] for item in recs_curr]) if recs_curr else 0.0
    exp_avg_score = np.mean([item["score"] for item in recs_exp]) if recs_exp else 0.0
    
    print(f"| {'Metric':<20} | {'Current':<10} | {'Experimental':<12} |", flush=True)
    print(f"| {'-'*20} | {'-'*10} | {'-'*12} |", flush=True)
    print(f"| {'Discover Eligible':<20} | {curr_len:<10} | {exp_len:<12} |", flush=True)
    print(f"| {'Unique Franchises':<20} | {curr_uniq:<10} | {exp_uniq:<12} |", flush=True)
    print(f"| {'Avg Similarity':<20} | {curr_avg_sim:<10.4f} | {exp_avg_sim:<12.4f} |", flush=True)
    print(f"| {'Avg Score':<20} | {curr_avg_score:<10.4f} | {exp_avg_score:<12.4f} |", flush=True)
    print(flush=True)

    # Step 8: Final Diagnosis
    print("### Final Diagnosis", flush=True)
    cg_500_contamination = sum(1 for cand_id, _, _ in cg_neighbors if get_franchise(service.catalog_meta[cand_id]["title"]) in cg_franchises or (service.catalog_meta[cand_id].get("title_english") and get_franchise(service.catalog_meta[cand_id]["title_english"]) in cg_franchises)) / 500
    
    cg_first_rank = 999
    for rank, (cand_id, _, _) in enumerate(cg_neighbors, 1):
        meta = service.catalog_meta[cand_id]
        title = meta["title"]
        eng_title = meta.get("title_english") or ""
        cand_f = get_franchise(title)
        cand_f_eng = get_franchise(eng_title) if eng_title else ""
        if cand_f in cg_franchises or (cand_f_eng and cand_f_eng in cg_franchises):
            continue
        root_id = service.get_franchise_root(cand_f)
        is_seq = (root_id is not None and cand_id != root_id) or service.is_sequel_title(title) or (eng_title and service.is_sequel_title(eng_title))
        if not is_seq:
            cg_first_rank = rank
            break

    if cg_500_contamination > 0.50 and cg_first_rank > 50:
        diagnosis = "ROOT CAUSE: EMBEDDING NEIGHBORHOOD CONTAMINATION"
        evidence = f"Code Geass contamination at Top 500 is {cg_500_contamination:.1%} (> 50%) and the first discoverable rank is at index {cg_first_rank} (> 50)."
    elif cg_first_rank < 20:
        diagnosis = "ROOT CAUSE: CANDIDATE ORDERING"
        evidence = f"Code Geass has a first discoverable rank at index {cg_first_rank} (< 20), indicating discoverable candidates are available early but out-ranked."
    else:
        diagnosis = "ROOT CAUSE: COMBINED EFFECT"
        evidence = f"Code Geass has a first discoverable rank at index {cg_first_rank} and Top 500 contamination of {cg_500_contamination:.1%}."

    print(diagnosis, flush=True)
    print(f"\nEvidence:\n - {evidence}\n", flush=True)

if __name__ == "__main__":
    main()
