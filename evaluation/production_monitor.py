import os
import sys
import json
import numpy as np
import pandas as pd
from collections import Counter

from pathlib import Path
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Force reranking enabled and 100% traffic for validation routing
os.environ["CINESENSE_RERANK_ENABLED"] = "True"
os.environ["CINESENSE_RERANK_TRAFFIC_PERCENT"] = "100"
# Configure to peak Model C configuration to align with the locked benchmark baseline (NDCG=0.2249)
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
    print("Initializing production monitoring suite...", flush=True)
    
    # 1. Load model and service
    model, catalog_df, _ = load_model(os.path.join(PROJECT_ROOT, "cinesense/models/twostage_v1"))
    service = RecommendationService(model, catalog_df)
    
    # Map title/english title to ID for resolving standard recommendations
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

    # Prepare seeds and ideal grades
    gold_seeds_mapped = []
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
            
        # Global IDCG
        ideal_grades = [2] * len(good_recs) + [1] * len(acc_recs)
        ideal_grades.sort(reverse=True)
        ideal_grades = ideal_grades[:10]
        while len(ideal_grades) < 10:
            ideal_grades.append(0)
        global_idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal_grades))
            
        gold_seeds_mapped.append({
            "seed_id": seed_id,
            "seed_title": entry["seed"],
            "good": good_recs,
            "acceptable": acc_recs,
            "global_idcg": global_idcg
        })

    # Evaluate all seeds and gather stats
    ndcgs, mrrs, precs = [], [], []
    discovery_rates = []
    franchise_diversities = []
    
    # Feature tracking lists
    semantic_scores_all = []
    jaccard_scores_all = []
    rerank_deltas_all = []
    all_recommended_ids = []
    
    print(f"Evaluating {len(gold_seeds_mapped)} benchmark seeds...", flush=True)
    
    for s_data in gold_seeds_mapped:
        seed_id = s_data["seed_id"]
        good_set = s_data["good"]
        acc_set = s_data["acceptable"]
        global_idcg = s_data["global_idcg"]
        
        # Call recommend using deterministic user_id routing to Treatment
        # Mode is "discover"
        recs = service.recommend([seed_id], mode="discover", user_id="validation_user_1", top_k=10)
        
        if not recs:
            continue
            
        rec_ids = [r["anime_id"] for r in recs]
        all_recommended_ids.extend(rec_ids)
        
        # Calculate standard recommendation quality metrics
        grades = [2 if a in good_set else (1 if a in acc_set else 0) for a in rec_ids]
        while len(grades) < 10:
            grades.append(0)
        ndcg, mrr, p10 = compute_metrics(grades, global_idcg=global_idcg)
        
        ndcgs.append(ndcg)
        mrrs.append(mrr)
        precs.append(p10)
        
        # Calculate discovery rate and franchise diversity
        seed_meta = service.catalog_meta.get(seed_id, {})
        seed_franchises = {get_franchise(seed_meta.get("title", ""))}
        if seed_meta.get("title_english"):
            seed_franchises.add(get_franchise(seed_meta["title_english"]))
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
        
        # Track score components using the _audit structure attached by recommend
        for r in recs:
            audit = r.get("_audit", {})
            sem_score = audit.get("semantic_score", r["score"]) # fallback if audit not present
            jaccard = audit.get("jaccard_similarity", 0.0)
            rerank_score = r["score"]
            delta = rerank_score - sem_score
            
            semantic_scores_all.append(sem_score)
            jaccard_scores_all.append(jaccard)
            rerank_deltas_all.append(delta)

    # Compute entropy of recommended IDs
    rec_counts = Counter(all_recommended_ids)
    total_recs = len(all_recommended_ids)
    entropy = -sum((count / total_recs) * np.log(count / total_recs) for count in rec_counts.values())

    # Compile final report values
    mean_ndcg = np.mean(ndcgs)
    mean_mrr = np.mean(mrrs)
    mean_prec = np.mean(precs)
    mean_dr = np.mean(discovery_rates)
    mean_fd = np.mean(franchise_diversities)
    
    mean_sem = np.mean(semantic_scores_all) if semantic_scores_all else 0.0
    mean_jac = np.mean(jaccard_scores_all) if jaccard_scores_all else 0.0
    mean_delta = np.mean(rerank_deltas_all) if rerank_deltas_all else 0.0
    
    print("\n" + "="*80)
    print("PRODUCTION MONITORING REPORT")
    print("="*80)
    print(f"Mean NDCG@10:             {mean_ndcg:.4f}")
    print(f"Mean MRR:                 {mean_mrr:.4f}")
    print(f"Mean Precision@10:        {mean_prec:.2%}")
    print(f"Mean Discovery Rate:      {mean_dr:.1f}%")
    print(f"Mean Franchise Diversity: {mean_fd:.2f}")
    print(f"Recommendation Entropy:   {entropy:.4f}")
    print(f"Mean Semantic Score:      {mean_sem:.4f}")
    print(f"Mean Jaccard Score:       {mean_jac:.4f}")
    print(f"Mean Rerank Delta:        {mean_delta:+.4f}")
    print("="*80)

    # Alert Conditions
    alert_triggered = False
    
    if mean_ndcg < 0.2024:
        print("ALERT: NDCG drop triggered! NDCG@10 is below 0.2024 target threshold.", file=sys.stderr)
        alert_triggered = True
    if mean_dr < 95.0:
        print("ALERT: Discovery rate is below 95.0% threshold.", file=sys.stderr)
        alert_triggered = True
    if mean_fd < 7.0:
        print("ALERT: Franchise diversity is below 7.0 threshold.", file=sys.stderr)
        alert_triggered = True

    if alert_triggered:
        print("\nSTATUS: FAILED", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nSTATUS: PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()
