import os
import sys
import json
import numpy as np
import pandas as pd

from pathlib import Path
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cinesense.utils.model_storage import load_model
from cinesense.services.recommendation import RecommendationService

def compute_metrics(recs, good_titles, acceptable_titles):
    # Determine relevance grades for top 10 recommendations
    relevance_grades = []
    
    good_set = {t.lower().strip() for t in good_titles}
    acceptable_set = {t.lower().strip() for t in acceptable_titles}
    
    for item in recs:
        title = item.get("title", "").lower().strip()
        title_eng = (item.get("title_english") or "").lower().strip()
        
        # Check if either title matches the gold standard sets
        is_good = (title in good_set) or (title_eng in good_set)
        is_acceptable = (title in acceptable_set) or (title_eng in acceptable_set)
        
        if is_good:
            relevance_grades.append(2)
        elif is_acceptable:
            relevance_grades.append(1)
        else:
            relevance_grades.append(0)
            
    # Pad to 10 if fewer recommendations returned
    while len(relevance_grades) < 10:
        relevance_grades.append(0)
        
    # 1. Precision@5
    p5 = sum(1 for r in relevance_grades[:5] if r > 0) / 5.0
    
    # 2. Precision@10
    p10 = sum(1 for r in relevance_grades[:10] if r > 0) / 10.0
    
    # 3. Recall@10
    total_relevant = len(good_set) + len(acceptable_set)
    r10 = sum(1 for r in relevance_grades[:10] if r > 0) / float(total_relevant) if total_relevant > 0 else 0.0
    
    # 4. NDCG@10
    dcg10 = 0.0
    for i, r in enumerate(relevance_grades[:10]):
        dcg10 += (2**r - 1) / np.log2(i + 2)
        
    # Ideal DCG@10 (sorted relevance grades of all possible relevant items)
    ideal_grades = [2] * len(good_set) + [1] * len(acceptable_set)
    ideal_grades.sort(reverse=True)
    ideal_grades = ideal_grades[:10]
    while len(ideal_grades) < 10:
        ideal_grades.append(0)
        
    idcg10 = 0.0
    for i, r in enumerate(ideal_grades[:10]):
        idcg10 += (2**r - 1) / np.log2(i + 2)
        
    ndcg10 = dcg10 / idcg10 if idcg10 > 0 else 0.0
    
    # 5. MRR
    mrr = 0.0
    for i, r in enumerate(relevance_grades[:10]):
        if r > 0:
            mrr = 1.0 / (i + 1)
            break
            
    return p5, p10, r10, ndcg10, mrr

def main():
    print("Loading model and gold standard dataset...", flush=True)
    model, catalog_df, metadata = load_model(os.path.join(PROJECT_ROOT, "cinesense/models/twostage_v1"))
    service = RecommendationService(model, catalog_df)

    # Enable O(1) franchise root caching
    original_get_franchise_root = service.get_franchise_root
    franchise_root_cache = {}
    def cached_get_franchise_root(franchise_name):
        if franchise_name not in franchise_root_cache:
            franchise_root_cache[franchise_name] = original_get_franchise_root(franchise_name)
        return franchise_root_cache[franchise_name]
    service.get_franchise_root = cached_get_franchise_root

    # Load gold standard dataset
    with open(os.path.join(PROJECT_ROOT, "evaluation/gold_standard_v2.json"), "r") as f:
        gold_dataset = json.load(f)
        
    print(f"Loaded {len(gold_dataset)} seeds. Running evaluation...", flush=True)

    # Make sure representation penalty is enabled for release candidate validation
    os.environ["CINESENSE_REPRESENTATION_PENALTY"] = "True"
    os.environ["CINESENSE_REPRESENTATION_LAMBDA"] = "0.03"

    rows = []
    
    for entry in gold_dataset:
        seed_name = entry["seed"]
        seed_id = entry["anime_id"]
        good_recs = entry["good_recommendations"]
        acc_recs = entry["acceptable_recommendations"]
        
        # Run discover recommendations
        recs = service.recommend([seed_id], ratings={seed_id: 10.0}, top_k=10, mode="discover")
        
        p5, p10, r10, ndcg10, mrr = compute_metrics(recs, good_recs, acc_recs)
        
        rows.append({
            "seed": seed_name,
            "P@5": p5,
            "P@10": p10,
            "R@10": r10,
            "NDCG@10": ndcg10,
            "MRR": mrr
        })
        
    df_metrics = pd.DataFrame(rows)
    
    print("\n" + "="*80)
    print("GOLD STANDARD EVALUATION PERFORMANCE PER SEED")
    print("="*80)
    print(f"| {'Seed':<40} | {'P@5':<6} | {'P@10':<6} | {'R@10':<6} | {'NDCG@10':<8} | {'MRR':<6} |")
    print(f"| {'-'*40} | {'-'*6} | {'-'*6} | {'-'*6} | {'-'*8} | {'-'*6} |")
    for _, r in df_metrics.iterrows():
        print(f"| {r['seed']:<40} | {r['P@5']:<6.2f} | {r['P@10']:<6.2f} | {r['R@10']:<6.2f} | {r['NDCG@10']:<8.4f} | {r['MRR']:<6.4f} |")
    print()
    
    print("="*80)
    print("AGGREGATE PERFORMANCE METRICS")
    print("="*80)
    print(f"* **Mean Precision@5:** {df_metrics['P@5'].mean():.2%}")
    print(f"* **Mean Precision@10:** {df_metrics['P@10'].mean():.2%}")
    print(f"* **Mean Recall@10:** {df_metrics['R@10'].mean():.2%}")
    print(f"* **Mean NDCG@10:** {df_metrics['NDCG@10'].mean():.4f}")
    print(f"* **Mean MRR:** {df_metrics['MRR'].mean():.4f}")
    print()

if __name__ == "__main__":
    main()
