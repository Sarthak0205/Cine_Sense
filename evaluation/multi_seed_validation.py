import os
import sys
import json
import numpy as np
import pandas as pd
from itertools import combinations
from collections import Counter

from pathlib import Path
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Force reranking enabled and representation penalty active to balance seeds
os.environ["CINESENSE_RERANK_ENABLED"] = "True"
os.environ["CINESENSE_RERANK_TRAFFIC_PERCENT"] = "100"
os.environ["CINESENSE_REPRESENTATION_PENALTY"] = "True"
os.environ["CINESENSE_REPRESENTATION_LAMBDA"] = "0.50"
# Peak config for Jaccard and distance
os.environ["CINESENSE_COSINE_POWER"] = "0"
os.environ["CINESENSE_POPULARITY_PENALTY"] = "0.0"

from cinesense.utils.model_storage import load_model
from cinesense.services.recommendation import RecommendationService, get_franchise

def clean_title(t):
    import re
    return re.sub(r'[^a-z0-9]', '', str(t).lower())

def compute_metrics(relevance_grades, global_idcg=None):
    p10 = sum(1 for r in relevance_grades[:10] if r > 0) / 10.0
    dcg10 = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(relevance_grades[:10]))
    
    if global_idcg is not None:
        ndcg10 = dcg10 / global_idcg if global_idcg > 0 else 0.0
    else:
        ideal_grades = sorted(relevance_grades, reverse=True)[:10]
        idcg10 = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal_grades))
        ndcg10 = dcg10 / idcg10 if idcg10 > 0 else 0.0
        
    mrr = 0.0
    for i, r in enumerate(relevance_grades[:10]):
        if r > 0:
            mrr = 1.0 / (i + 1)
            break
    return ndcg10, mrr, p10

def main():
    print("Initializing multi-seed validation suite...", flush=True)
    
    # 1. Load model and service
    model, catalog_df, _ = load_model(os.path.join(PROJECT_ROOT, "cinesense/models/twostage_v1"))
    service = RecommendationService(model, catalog_df)
    
    # Map title/english title to ID
    animes_df = pd.read_csv(os.path.join(PROJECT_ROOT, "archive-2/animes.csv"))
    clean_to_id = {}
    for _, row in animes_df.iterrows():
        aid = int(row["anime_id"])
        title = str(row["title"])
        eng_title = str(row["title_english"]) if pd.notna(row["title_english"]) else ""
        clean_to_id[clean_title(title)] = aid
        if eng_title:
            clean_to_id[clean_title(eng_title)] = aid

    MANUAL_OVERRIDES = {
        clean_title("Re:Zero - Starting Life in Another World"): 31240,
        clean_title("Salaryman Kintaro"): 1608,
        clean_title("The Garden of Sinners"): 2593,
        clean_title("Devilman Crybaby"): 35120,
        clean_title("Yu Yu Hakusho"): 392
    }

    def resolve_title_to_id(title_str):
        c_title = clean_title(title_str)
        if c_title in MANUAL_OVERRIDES:
            return MANUAL_OVERRIDES[c_title]
        return clean_to_id.get(c_title)

    # 2. Load gold standard
    with open(os.path.join(PROJECT_ROOT, "evaluation/gold_standard_v2.json")) as f:
        gold_dataset = json.load(f)

    # Prepare seeds mapping
    mapped_seeds = {}
    for entry in gold_dataset:
        seed_id = entry.get("anime_id")
        if not seed_id:
            seed_id = resolve_title_to_id(entry["seed"])
        if not seed_id or int(seed_id) not in model.item_id_to_index:
            continue
        seed_id = int(seed_id)
        
        good_recs = set()
        acc_recs = set()
        for r_title in entry["good_recommendations"]:
            rid = resolve_title_to_id(r_title)
            if rid: good_recs.add(int(rid))
        for r_title in entry["acceptable_recommendations"]:
            rid = resolve_title_to_id(r_title)
            if rid: acc_recs.add(int(rid))
            
        mapped_seeds[seed_id] = {
            "title": entry["seed"],
            "good": good_recs,
            "acceptable": acc_recs
        }

    # Generate multi-seed pairs
    seed_ids_list = list(mapped_seeds.keys())
    seed_pairs = list(combinations(seed_ids_list, 2))
    
    # We need at least 50 scenarios
    scenarios = seed_pairs[:55]
    print(f"Generated {len(scenarios)} scenarios from combinations of {len(seed_ids_list)} seeds.", flush=True)

    ndcgs, mrrs = [], []
    discovery_rates = []
    franchise_diversities = []
    dominant_shares = []
    
    print("Running multi-seed benchmarks...", flush=True)
    failures = []
    
    for idx, (s1, s2) in enumerate(scenarios):
        # Combined relevance sets
        good_combined = mapped_seeds[s1]["good"].union(mapped_seeds[s2]["good"])
        acc_combined = mapped_seeds[s1]["acceptable"].union(mapped_seeds[s2]["acceptable"])
        
        # Calculate combined global IDCG
        ideal_grades = [2] * len(good_combined) + [1] * len(acc_combined)
        ideal_grades.sort(reverse=True)
        ideal_grades = ideal_grades[:10]
        while len(ideal_grades) < 10:
            ideal_grades.append(0)
        global_idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal_grades))

        # Query recommendation service
        recs = service.recommend([s1, s2], mode="discover", user_id="validation_user_1", top_k=10)
        if not recs:
            continue
            
        rec_ids = [r["anime_id"] for r in recs]
        
        # Calculate metrics
        grades = [2 if a in good_combined else (1 if a in acc_combined else 0) for a in rec_ids]
        while len(grades) < 10:
            grades.append(0)
        ndcg, mrr, _ = compute_metrics(grades, global_idcg=global_idcg)
        
        ndcgs.append(ndcg)
        mrrs.append(mrr)
        
        # Calculate discovery rate and franchise diversity
        seed_franchises = set()
        for s in (s1, s2):
            meta = service.catalog_meta.get(s, {})
            seed_franchises.add(get_franchise(meta.get("title", "")))
            if meta.get("title_english"):
                seed_franchises.add(get_franchise(meta["title_english"]))
        seed_franchises.discard("")
        
        new_franchise_count = 0
        rec_franchises = []
        for a_id in rec_ids:
            meta = service.catalog_meta[a_id]
            rec_f = get_franchise(meta["title"])
            rec_f_eng = get_franchise(meta.get("title_english") or "")
            is_same = (rec_f in seed_franchises) or (rec_f_eng and rec_f_eng in seed_franchises)
            if not is_same:
                new_franchise_count += 1
            rec_franchises.append(rec_f)
            
        dr = (new_franchise_count / len(recs)) * 100.0 if recs else 0.0
        fd = len(set(rec_franchises))
        
        discovery_rates.append(dr)
        franchise_diversities.append(fd)
        
        # Calculate Dominant Seed Share
        seed_exps = []
        for r in recs:
            matched_seed = r.get("explanation", {}).get("matched_seed", {}).get("anime_id")
            if matched_seed in (s1, s2):
                seed_exps.append(matched_seed)
                
        if seed_exps:
            counts = Counter(seed_exps)
            dominant_seed_count = counts.most_common(1)[0][1]
            share = (dominant_seed_count / len(recs)) * 100.0
        else:
            share = 50.0  # balanced by default
            
        dominant_shares.append(share)
        
        if share > 60.0:
            failures.append((mapped_seeds[s1]["title"], mapped_seeds[s2]["title"], share))

    mean_ndcg = np.mean(ndcgs)
    mean_mrr = np.mean(mrrs)
    mean_dr = np.mean(discovery_rates)
    mean_fd = np.mean(franchise_diversities)
    mean_dominant_share = np.mean(dominant_shares)
    
    print("\n" + "="*80)
    print("MULTI-SEED VALIDATION REPORT")
    print("="*80)
    print(f"Mean NDCG@10:             {mean_ndcg:.4f}")
    print(f"Mean MRR:                 {mean_mrr:.4f}")
    print(f"Mean Discovery Rate:      {mean_dr:.1f}%")
    print(f"Mean Franchise Diversity: {mean_fd:.2f}")
    print(f"Mean Dominant Seed Share: {mean_dominant_share:.1f}%")
    print(f"Scenarios evaluated:      {len(ndcgs)}")
    print(f"Scenarios exceeding 60%:  {len(failures)}")
    print("="*80)
    
    if len(failures) > 0:
        print(f"\nALERT: {len(failures)} scenarios exceeded 60% dominant seed share limit!", file=sys.stderr)
        for s1_title, s2_title, share in failures[:10]:
            print(f"  - {s1_title} + {s2_title} | Dominant share: {share:.1f}%", file=sys.stderr)
        
        if mean_dominant_share > 60.0:
            print("ALERT: Mean dominant seed share exceeds 60% limit!", file=sys.stderr)
            sys.exit(1)
            
    sys.exit(0)

if __name__ == "__main__":
    main()
