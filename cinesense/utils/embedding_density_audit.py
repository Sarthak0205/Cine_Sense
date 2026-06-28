import os
import sys
import re
import numpy as np
import pandas as pd

# Set PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cinesense.utils.model_storage import load_model
from cinesense.services.recommendation import RecommendationService, get_franchise
from cinesense.retrieval.hybrid_c import hybrid_c_retrieval_scores, top_retrieval_indices

def main():
    print("Loading model for embedding density audit...", flush=True)
    model, catalog_df, metadata = load_model(os.path.join(PROJECT_ROOT, "cinesense/models/twostage_v1"))
    service = RecommendationService(model, catalog_df)

    # Cache get_franchise_root to keep total runtimes low
    original_get_franchise_root = service.get_franchise_root
    franchise_root_cache = {}
    def cached_get_franchise_root(franchise_name):
        if franchise_name not in franchise_root_cache:
            franchise_root_cache[franchise_name] = original_get_franchise_root(franchise_name)
        return franchise_root_cache[franchise_name]
    service.get_franchise_root = cached_get_franchise_root

    seeds_config = {
        1535: "Death Note",
        1575: "Code Geass",
        9253: "Steins;Gate",
        5114: "FMAB",
        16498: "Attack on Titan",
        11061: "Hunter x Hunter",
        21: "One Piece",
        20: "Naruto",
        269: "Bleach",
        19: "Monster",
        32379: "Berserk",
        5680: "K-On",
        30: "Evangelion"
    }

    # ----------------------------------------------------
    # Part 1: Density Metrics
    # ----------------------------------------------------
    print("\n" + "="*80)
    print("PART 1: DENSITY METRICS")
    print("="*80)
    
    density_stats = {}
    for sid, name in seeds_config.items():
        idx_s = model.item_id_to_index[sid]
        emb_s = model.catalog_embeddings[idx_s]
        
        sims = []
        for i in range(len(model.catalog_embeddings)):
            if i == idx_s:
                continue
            sim = float(np.dot(emb_s, model.catalog_embeddings[i]))
            sims.append(sim)
        sims.sort(reverse=True)
        
        density_stats[sid] = {}
        for k in [50, 100, 300]:
            sub = sims[:k]
            avg_sim = np.mean(sub)
            med_sim = np.median(sub)
            std_sim = np.std(sub)
            max_sim = np.max(sub)
            min_sim = np.min(sub)
            density_stats[sid][k] = {
                "avg": avg_sim,
                "median": med_sim,
                "std": std_sim,
                "max": max_sim,
                "min": min_sim
            }
            
        print(f"\nSeed: {name} (ID: {sid})")
        for k in [50, 100, 300]:
            s = density_stats[sid][k]
            print(f"  Top {k:<3} -> Avg: {s['avg']:.4f} | Median: {s['median']:.4f} | Std: {s['std']:.4f} | Max: {s['max']:.4f} | Min: {s['min']:.4f}")

    # ----------------------------------------------------
    # Part 2: Density Ranking
    # ----------------------------------------------------
    print("\n" + "="*80)
    print("PART 2: DENSITY RANKING")
    print("="*80)
    
    ranked_list = []
    for sid, name in seeds_config.items():
        ranked_list.append({
            "name": name,
            "avg100": density_stats[sid][100]["avg"],
            "avg300": density_stats[sid][300]["avg"]
        })
        
    # Sort by AvgSim100 descending
    ranked_list.sort(key=lambda x: -x["avg100"])
    
    print(f"| {'Rank':<4} | {'Seed':<35} | {'AvgSim100':<10} | {'AvgSim300':<10} |")
    print(f"| {'-'*4} | {'-'*35} | {'-'*10} | {'-'*10} |")
    for r_idx, item in enumerate(ranked_list, 1):
        print(f"| {r_idx:<4} | {item['name']:<35} | {item['avg100']:<10.4f} | {item['avg300']:<10.4f} |")

    # ----------------------------------------------------
    # Part 3: Neighbor Quality
    # ----------------------------------------------------
    print("\n" + "="*80)
    print("PART 3: NEIGHBOR QUALITY (Top 100 Neighbors)")
    print("="*80)
    
    keywords = ["movie", "film", "ova", "ona", "special", "specials", "recap", "pilot"]
    def is_movie_ova_special(t, eng_t):
        t_low = (t or "").lower()
        eng_low = (eng_t or "").lower()
        return any(kw in t_low or kw in eng_low for kw in keywords)

    print(f"| {'Seed':<25} | {'Discover Eligible %':<20} | {'Same Franchise %':<17} | {'Sequel %':<9} | {'Movie/OVA %':<11} |")
    print(f"| {'-'*25} | {'-'*20} | {'-'*17} | {'-'*9} | {'-'*11} |")

    for sid, name in seeds_config.items():
        idx_s = model.item_id_to_index[sid]
        emb_s = model.catalog_embeddings[idx_s]
        
        sims = []
        for i in range(len(model.catalog_embeddings)):
            if i == idx_s:
                continue
            sim = float(np.dot(emb_s, model.catalog_embeddings[i]))
            sims.append((int(model.anime_ids[i]), sim))
        sims.sort(key=lambda x: -x[1])
        
        seed_meta = service.catalog_meta[sid]
        seed_franchises = {get_franchise(seed_meta["title"])}
        if seed_meta.get("title_english"):
            seed_franchises.add(get_franchise(seed_meta["title_english"]))
            
        sub = sims[:100]
        same_f = 0
        sequel = 0
        movie_ova = 0
        duplicate_f = 0
        discover_elg = 0
        
        seen_franchises = set()
        for item_id, sim in sub:
            meta = service.catalog_meta[item_id]
            title = meta["title"]
            eng_title = meta.get("title_english") or ""
            
            cand_f = get_franchise(title)
            cand_f_eng = get_franchise(eng_title) if eng_title else ""
            
            if cand_f in seed_franchises or (cand_f_eng and cand_f_eng in seed_franchises):
                same_f += 1
                continue
                
            root_id = service.get_franchise_root(cand_f)
            is_seq = False
            if root_id is not None and item_id != root_id:
                is_seq = True
            elif service.is_sequel_title(title) or (eng_title and service.is_sequel_title(eng_title)):
                is_seq = True
                
            if is_seq:
                sequel += 1
                continue
                
            if is_movie_ova_special(title, eng_title):
                movie_ova += 1
                continue
                
            if cand_f in seen_franchises or (cand_f_eng and cand_f_eng in seen_franchises):
                duplicate_f += 1
            else:
                discover_elg += 1
                seen_franchises.add(cand_f)
                if cand_f_eng:
                    seen_franchises.add(cand_f_eng)
                    
        print(f"| {name:<25} | {discover_elg/100.0:<20.1%} | {same_f/100.0:<17.1%} | {sequel/100.0:<9.1%} | {movie_ova/100.0:<11.1%} |")

    # ----------------------------------------------------
    # Part 4: Multi-Seed Survivability
    # ----------------------------------------------------
    print("\n" + "="*80)
    print("PART 4: MULTI-SEED RETRIEVED ATTRIBUTION SHARE")
    print("="*80)
    
    print(f"| {'Seed':<25} | {'DN + Seed Share':<17} | {'DN + SG + Seed Share':<20} | {'AoT + DN + Seed Share':<21} |")
    print(f"| {'-'*25} | {'-'*17} | {'-'*20} | {'-'*21} |")

    for sid, name in seeds_config.items():
        # Scenarios setup
        # 1. DN + Seed
        if sid == 1535: # DN itself
            s1_share_str = "N/A"
            s2_share_str = "N/A"
            s3_share_str = "N/A"
        else:
            # S1: DN + Seed
            s1_seeds = [1535, sid]
            s1_ratings = {1535: 10.0, sid: 9.0}
            train_indices = np.asarray([model.item_id_to_index[aid] for aid in s1_seeds], dtype=np.int32)
            train_weights = np.asarray([model._rating_weight(int(s1_ratings[aid])) for aid in s1_seeds], dtype=np.float32)
            retrieval_scores = hybrid_c_retrieval_scores(
                train_indices, model.catalog_embeddings, model.popularity_scores,
                model.semantic_weight, model.popularity_weight, model.seed_batch_size
            )
            retrieved_indices_raw = top_retrieval_indices(retrieval_scores, set(s1_seeds), model.anime_ids, 300)
            
            # Count attributes
            s1_embeddings = {aid: model.catalog_embeddings[model.item_id_to_index[aid]] for aid in s1_seeds}
            s1_counts = {aid: 0 for aid in s1_seeds}
            for idx in retrieved_indices_raw:
                emb_c = model.catalog_embeddings[idx]
                best_seed = None
                max_sim = -float('inf')
                for aid, emb_s in s1_embeddings.items():
                    sim = float(np.dot(emb_c, emb_s))
                    if sim > max_sim:
                        max_sim = sim
                        best_seed = aid
                if best_seed in s1_counts:
                    s1_counts[best_seed] += 1
            s1_share_str = f"{s1_counts[sid] / 300.0:.1%}"
            
            # S2: DN + SG + Seed
            if sid == 9253: # SG itself
                s2_share_str = "N/A"
            else:
                s2_seeds = [1535, 9253, sid]
                s2_ratings = {1535: 10.0, 9253: 9.0, sid: 8.0}
                train_indices = np.asarray([model.item_id_to_index[aid] for aid in s2_seeds], dtype=np.int32)
                train_weights = np.asarray([model._rating_weight(int(s2_ratings[aid])) for aid in s2_seeds], dtype=np.float32)
                retrieval_scores = hybrid_c_retrieval_scores(
                    train_indices, model.catalog_embeddings, model.popularity_scores,
                    model.semantic_weight, model.popularity_weight, model.seed_batch_size
                )
                retrieved_indices_raw = top_retrieval_indices(retrieval_scores, set(s2_seeds), model.anime_ids, 300)
                
                s2_embeddings = {aid: model.catalog_embeddings[model.item_id_to_index[aid]] for aid in s2_seeds}
                s2_counts = {aid: 0 for aid in s2_seeds}
                for idx in retrieved_indices_raw:
                    emb_c = model.catalog_embeddings[idx]
                    best_seed = None
                    max_sim = -float('inf')
                    for aid, emb_s in s2_embeddings.items():
                        sim = float(np.dot(emb_c, emb_s))
                        if sim > max_sim:
                            max_sim = sim
                            best_seed = aid
                    if best_seed in s2_counts:
                        s2_counts[best_seed] += 1
                s2_share_str = f"{s2_counts[sid] / 300.0:.1%}"
                
            # S3: AoT + DN + Seed
            if sid == 16498: # AoT itself
                s3_share_str = "N/A"
            else:
                s3_seeds = [16498, 1535, sid]
                s3_ratings = {16498: 10.0, 1535: 9.0, sid: 8.0}
                train_indices = np.asarray([model.item_id_to_index[aid] for aid in s3_seeds], dtype=np.int32)
                train_weights = np.asarray([model._rating_weight(int(s3_ratings[aid])) for aid in s3_seeds], dtype=np.float32)
                retrieval_scores = hybrid_c_retrieval_scores(
                    train_indices, model.catalog_embeddings, model.popularity_scores,
                    model.semantic_weight, model.popularity_weight, model.seed_batch_size
                )
                retrieved_indices_raw = top_retrieval_indices(retrieval_scores, set(s3_seeds), model.anime_ids, 300)
                
                s3_embeddings = {aid: model.catalog_embeddings[model.item_id_to_index[aid]] for aid in s3_seeds}
                s3_counts = {aid: 0 for aid in s3_seeds}
                for idx in retrieved_indices_raw:
                    emb_c = model.catalog_embeddings[idx]
                    best_seed = None
                    max_sim = -float('inf')
                    for aid, emb_s in s3_embeddings.items():
                        sim = float(np.dot(emb_c, emb_s))
                        if sim > max_sim:
                            max_sim = sim
                            best_seed = aid
                    if best_seed in s3_counts:
                        s3_counts[best_seed] += 1
                s3_share_str = f"{s3_counts[sid] / 300.0:.1%}"
                
        print(f"| {name:<25} | {s1_share_str:<17} | {s2_share_str:<20} | {s3_share_str:<21} |")

    # ----------------------------------------------------
    # Part 5: Cluster Health Classification
    # ----------------------------------------------------
    print("\n" + "="*80)
    print("PART 5: CLUSTER HEALTH CLASSIFICATION")
    print("="*80)
    
    # Calculate thresholds from observed avg100 values
    all_avg100s = [density_stats[sid][100]["avg"] for sid in seeds_config.keys()]
    min_val = min(all_avg100s)
    max_val = max(all_avg100s)
    span = max_val - min_val
    
    # Bucket into 5 ranges
    # A >= max - 0.2*span
    # B >= max - 0.4*span
    # C >= max - 0.6*span
    # D >= max - 0.8*span
    # E < max - 0.8*span
    t_a = max_val - 0.2 * span
    t_b = max_val - 0.4 * span
    t_c = max_val - 0.6 * span
    t_d = max_val - 0.8 * span
    
    print(f"Observed Top 100 Avg Similarity Range: {min_val:.4f} to {max_val:.4f}")
    print(f"Classification Thresholds (Top 100 Avg Sim):")
    print(f"  - A (Very Dense):       >= {t_a:.4f}")
    print(f"  - B (Dense):            {t_b:.4f} to {t_a:.4f}")
    print(f"  - C (Average):          {t_c:.4f} to {t_b:.4f}")
    print(f"  - D (Sparse):           {t_d:.4f} to {t_c:.4f}")
    print(f"  - E (Extremely Sparse): < {t_d:.4f}")
    print()

    print(f"| {'Seed':<25} | {'AvgSim100':<10} | {'Class':<5} | {'Cluster Health Description':<25} |")
    print(f"| {'-'*25} | {'-'*10} | {'-'*5} | {'-'*25} |")
    for sid, name in seeds_config.items():
        avg_val = density_stats[sid][100]["avg"]
        if avg_val >= t_a:
            cls = "A"
            desc = "Very Dense (Monopolizes)"
        elif avg_val >= t_b:
            cls = "B"
            desc = "Dense (Strong)"
        elif avg_val >= t_c:
            cls = "C"
            desc = "Average (Balanced)"
        elif avg_val >= t_d:
            cls = "D"
            desc = "Sparse (Suppressed)"
        else:
            cls = "E"
            desc = "Extremely Sparse (0% Share)"
        print(f"| {name:<25} | {avg_val:<10.4f} | {cls:<5} | {desc:<25} |")
        
    print()

if __name__ == "__main__":
    main()
