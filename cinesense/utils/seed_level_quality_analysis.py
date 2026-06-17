import os
import sys
import json
import numpy as np
import pandas as pd

# Set PYTHONPATH
PROJECT_ROOT = "/Users/sdc/Projects/CineSense-main"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cinesense.utils.model_storage import load_model
from cinesense.services.recommendation import RecommendationService

# Category Mapping
seed_categories = {
    "Death Note": ["Psychological", "Thriller", "Mainstream"],
    "Code Geass: Lelouch of the Rebellion": ["Sci-Fi", "Psychological", "Mainstream"],
    "Steins;Gate": ["Sci-Fi", "Thriller", "Mainstream"],
    "Attack on Titan": ["Action", "Fantasy", "Mainstream"],
    "Fullmetal Alchemist: Brotherhood": ["Fantasy", "Action", "Mainstream"],
    "Hunter x Hunter": ["Fantasy", "Action", "Mainstream"],
    "One Piece": ["Fantasy", "Action", "Mainstream"],
    "Naruto": ["Fantasy", "Action", "Mainstream"],
    "Bleach": ["Fantasy", "Action", "Mainstream"],
    "Monster": ["Thriller", "Psychological", "Niche"],
    "Berserk": ["Action", "Fantasy", "Niche"],
    "K-On!": ["Slice of Life", "Comedy", "Niche"],
    "Neon Genesis Evangelion": ["Sci-Fi", "Psychological", "Niche"],
    "Cowboy Bebop": ["Sci-Fi", "Action", "Niche"],
    "Psycho-Pass": ["Sci-Fi", "Thriller", "Mainstream"],
    "Fate/Zero": ["Fantasy", "Action", "Mainstream"],
    "Haikyu!!": ["Sports", "Comedy", "Mainstream"],
    "Gintama": ["Comedy", "Action", "Mainstream"],
    "Konosuba": ["Comedy", "Fantasy", "Mainstream"],
    "Clannad": ["Slice of Life", "Mainstream"],
    "Violet Evergarden": ["Slice of Life", "Mainstream"],
    "Erased": ["Thriller", "Psychological", "Mainstream"],
    "Parasyte -the maxim-": ["Thriller", "Action", "Mainstream"],
    "From the New World": ["Psychological", "Fantasy", "Niche"],
    "Your Lie in April": ["Slice of Life", "Mainstream"],
    "Anohana: The Flower We Saw That Day": ["Slice of Life", "Mainstream"],
    "Serial Experiments Lain": ["Sci-Fi", "Psychological", "Niche"],
    "Ergo Proxy": ["Sci-Fi", "Psychological", "Niche"],
    "Ping Pong the Animation": ["Sports", "Niche"],
    "Great Teacher Onizuka": ["Comedy", "Slice of Life", "Niche"],
    "Mob Psycho 100": ["Comedy", "Action", "Mainstream"],
    "Mushishi": ["Slice of Life", "Niche"],
    "Puella Magi Madoka Magica": ["Psychological", "Fantasy", "Niche"],
    "Toradora!": ["Comedy", "Slice of Life", "Mainstream"],
    "Black Lagoon": ["Action", "Niche"]
}

def compute_metrics(recs, good_titles, acceptable_titles):
    relevance_grades = []
    good_set = {t.lower().strip() for t in good_titles}
    acceptable_set = {t.lower().strip() for t in acceptable_titles}
    
    for item in recs:
        title = item.get("title", "").lower().strip()
        title_eng = (item.get("title_english") or "").lower().strip()
        
        is_good = (title in good_set) or (title_eng in good_set)
        is_acceptable = (title in acceptable_set) or (title_eng in acceptable_set)
        
        if is_good:
            relevance_grades.append(2)
        elif is_acceptable:
            relevance_grades.append(1)
        else:
            relevance_grades.append(0)
            
    while len(relevance_grades) < 10:
        relevance_grades.append(0)
        
    p5 = sum(1 for r in relevance_grades[:5] if r > 0) / 5.0
    p10 = sum(1 for r in relevance_grades[:10] if r > 0) / 10.0
    
    total_relevant = len(good_set) + len(acceptable_set)
    r10 = sum(1 for r in relevance_grades[:10] if r > 0) / float(total_relevant) if total_relevant > 0 else 0.0
    
    dcg10 = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(relevance_grades[:10]))
        
    ideal_grades = [2] * len(good_set) + [1] * len(acceptable_set)
    ideal_grades.sort(reverse=True)
    ideal_grades = ideal_grades[:10]
    while len(ideal_grades) < 10:
        ideal_grades.append(0)
        
    idcg10 = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal_grades[:10]))
    ndcg10 = dcg10 / idcg10 if idcg10 > 0 else 0.0
    
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

    # Force baseline production settings
    os.environ["CINESENSE_REPRESENTATION_PENALTY"] = "True"
    os.environ["CINESENSE_REPRESENTATION_LAMBDA"] = "0.03"

    rows = []
    
    print(f"Running evaluation on {len(gold_dataset)} seeds...", flush=True)
    for entry in gold_dataset:
        seed_name = entry["seed"]
        seed_id = entry["anime_id"]
        good_recs = entry["good_recommendations"]
        acc_recs = entry["acceptable_recommendations"]
        
        # Run baseline discover recommendations
        recs = service.recommend([seed_id], ratings={seed_id: 10.0}, top_k=10, mode="discover")
        p5, p10, r10, ndcg10, mrr = compute_metrics(recs, good_recs, acc_recs)
        
        rows.append({
            "seed": seed_name,
            "P5": p5,
            "P10": p10,
            "R10": r10,
            "NDCG10": ndcg10,
            "MRR": mrr
        })
        
    df_metrics = pd.DataFrame(rows)
    
    # Sort seeds by NDCG10 descending
    df_sorted = df_metrics.sort_values(by="NDCG10", ascending=False).reset_index(drop=True)
    
    # Print Seed Ranking Table
    print("\n" + "="*80)
    print("SEED RANKING BY NDCG@10")
    print("="*80)
    print(f"| {'Rank':<4} | {'Seed':<40} | {'P@10':<6} | {'Recall@10':<9} | {'NDCG@10':<8} | {'MRR':<6} |")
    print(f"| {'-'*4} | {'-'*40} | {'-'*6} | {'-'*9} | {'-'*8} | {'-'*6} |")
    for r_idx, r in df_sorted.iterrows():
        print(f"| {r_idx+1:<4} | {r['seed']:<40} | {r['P10']:<6.2%} | {r['R10']:<9.2%} | {r['NDCG10']:<8.4f} | {r['MRR']:<6.4f} |")
    print()

    # Top 10 Best and Worst
    print("TOP 10 BEST-PERFORMING SEEDS:")
    for idx, r in df_sorted.head(10).iterrows():
        print(f"  {idx+1}. {r['seed']} (NDCG: {r['NDCG10']:.4f})")
    print()
    
    print("TOP 10 WORST-PERFORMING SEEDS:")
    for idx, r in df_sorted.tail(10).iterrows():
        print(f"  {idx+1}. {r['seed']} (NDCG: {r['NDCG10']:.4f})")
    print()

    # ----------------------------------------------------
    # Category Grouping and Averages
    # ----------------------------------------------------
    # We map each seed to its categories
    category_rows = []
    for _, r in df_metrics.iterrows():
        name = r["seed"]
        cats = seed_categories.get(name, ["Other"])
        for cat in cats:
            category_rows.append({
                "category": cat,
                "P10": r["P10"],
                "R10": r["R10"],
                "NDCG10": r["NDCG10"],
                "MRR": r["MRR"]
            })
            
    df_cat = pd.DataFrame(category_rows)
    cat_summary = df_cat.groupby("category").mean().reset_index()
    cat_summary = cat_summary.sort_values(by="NDCG10", ascending=False).reset_index(drop=True)
    
    print("="*80)
    print("CATEGORY LEVEL AVERAGES")
    print("="*80)
    print(f"| {'Category':<15} | {'P@10':<6} | {'Recall@10':<9} | {'NDCG@10':<8} | {'MRR':<6} |")
    print(f"| {'-'*15} | {'-'*6} | {'-'*9} | {'-'*8} | {'-'*6} |")
    for _, r in cat_summary.iterrows():
        print(f"| {r['category']:<15} | {r['P10']:<6.2%} | {r['R10']:<9.2%} | {r['NDCG10']:<8.4f} | {r['MRR']:<6.4f} |")
    print()

if __name__ == "__main__":
    main()
