import os
import sys
import numpy as np
import pandas as pd

# Ensure cinesense and api are importable
sys.path.insert(0, os.path.abspath("."))

from cinesense.utils.model_storage import load_model
from cinesense.services.recommendation import RecommendationService, get_franchise

def main():
    model_dir = "cinesense/models/twostage_v1"
    print(f"Loading model from {model_dir}...", flush=True)
    model, catalog_df, metadata = load_model(model_dir)
    service = RecommendationService(model, catalog_df)
    
    # Calculate popularity percentiles over the entire catalog
    popularity_scores = model.popularity_scores
    sorted_pop = np.sort(popularity_scores)[::-1] # descending order
    total_items = len(sorted_pop)
    
    def get_pop_percentile_class(pop_score):
        # Find the rank of the pop_score in sorted_pop
        # Because there could be duplicate values, we find the first index where sorted_pop <= pop_score
        indices = np.where(sorted_pop <= pop_score)[0]
        if len(indices) == 0:
            rank = total_items
        else:
            rank = indices[0] + 1
        
        ratio = rank / total_items
        if ratio <= 0.01:
            return "Top 1%"
        elif ratio <= 0.05:
            return "Top 5%"
        elif ratio <= 0.10:
            return "Top 10%"
        else:
            return "Rest of catalog"
            
    scenarios = {
        "Scenario A": {
            "anime_ids": [1535],
            "ratings": {1535: 10.0},
            "top_k": 10,
            "mode": "discover",
            "interpretation": "Death Note superfan"
        },
        "Scenario B": {
            "anime_ids": [1575],
            "ratings": {1575: 10.0},
            "top_k": 10,
            "mode": "discover",
            "interpretation": "Code Geass fan"
        },
        "Scenario C": {
            "anime_ids": [9253],
            "ratings": {9253: 10.0},
            "top_k": 10,
            "mode": "discover",
            "interpretation": "Steins;Gate fan"
        },
        "Scenario D": {
            "anime_ids": [16498],
            "ratings": {16498: 10.0},
            "top_k": 10,
            "mode": "discover",
            "interpretation": "Attack on Titan fan"
        },
        "Scenario E": {
            "anime_ids": [5114],
            "ratings": {5114: 10.0},
            "top_k": 10,
            "mode": "discover",
            "interpretation": "Fullmetal Alchemist: Brotherhood fan"
        },
        "Scenario F": {
            "anime_ids": [1535, 1575, 9253],
            "ratings": {1535: 10.0, 1575: 9.0, 9253: 8.0},
            "top_k": 10,
            "mode": "discover",
            "interpretation": "Death Note (10.0), Code Geass (9.0), Steins;Gate (8.0)"
        }
    }
    
    print("\n=== RUNNING RECOMMENDATION QUALITY AUDIT ===", flush=True)
    
    for s_name, config in scenarios.items():
        print(f"\n==================================================")
        print(f"{s_name} - {config['interpretation']}")
        print(f"==================================================")
        
        # Get recommendations
        recs = service.recommend(
            anime_ids=config["anime_ids"],
            ratings=config["ratings"],
            top_k=config["top_k"],
            mode=config["mode"]
        )
        
        # We also need the seed franchises to check discovery rate
        seed_franchises = set()
        for aid in config["anime_ids"]:
            meta = service.catalog_meta.get(aid)
            if meta:
                seed_franchises.add(get_franchise(meta["title"]))
                if meta.get("title_english"):
                    seed_franchises.add(get_franchise(meta["title_english"]))
        
        rec_franchises = []
        pop_classes = []
        same_franchise_count = 0
        new_franchise_count = 0
        
        print(f"{'Rank':<5} | {'ID':<6} | {'Title':<45} | {'Score':<6} | {'Matched Seed':<30} | {'Sim':<6} | {'Pop':<6} | {'Percentile':<15} | {'Franchise':<30}")
        print("-" * 170)
        
        for idx, rec in enumerate(recs):
            aid = rec["anime_id"]
            title = rec["title"]
            title_eng = rec["title_english"] or ""
            score = rec["score"]
            explanation = rec["explanation"]
            matched_seed_title = explanation["matched_seed"]["title"]
            similarity = explanation["similarity"]
            popularity = explanation["popularity"]
            
            pop_class = get_pop_percentile_class(popularity)
            pop_classes.append(pop_class)
            
            # Determine franchise of recommendation
            rec_f_title = get_franchise(title)
            rec_f_eng = get_franchise(title_eng) if title_eng else ""
            
            # Franchise categorization
            # We will use rec_f_title for printing but let's check both for same seed franchise
            is_same_franchise = (rec_f_title in seed_franchises) or (rec_f_eng in seed_franchises)
            if is_same_franchise:
                same_franchise_count += 1
                classification = "Same franchise"
            else:
                new_franchise_count += 1
                classification = "New franchise"
                
            rec_franchises.append(rec_f_title)
            
            # Truncate titles for formatting
            disp_title = title[:42] + "..." if len(title) > 45 else title
            disp_seed = matched_seed_title[:27] + "..." if len(matched_seed_title) > 30 else matched_seed_title
            
            print(f"{idx+1:<5} | {aid:<6} | {disp_title:<45} | {score:<6.4f} | {disp_seed:<30} | {similarity:<6.4f} | {popularity:<6.4f} | {pop_class:<15} | {rec_f_title:<30} ({classification})")
            
        # Metrics calculation
        unique_franchises = list(set(rec_franchises))
        num_unique = len(unique_franchises)
        num_duplicates = len(rec_franchises) - num_unique
        dup_ratio = (num_duplicates / len(rec_franchises)) * 100 if rec_franchises else 0
        discovery_rate = (new_franchise_count / len(recs)) * 100 if recs else 0
        
        pop_dist = {
            "Top 1%": pop_classes.count("Top 1%"),
            "Top 5%": pop_classes.count("Top 5%"),
            "Top 10%": pop_classes.count("Top 10%"),
            "Rest of catalog": pop_classes.count("Rest of catalog")
        }
        
        print("\n--- SUMMARY METRICS ---")
        print(f"Franchise Metrics:")
        print(f"  - Unique Franchises: {num_unique}")
        print(f"  - Duplicate Franchises: {num_duplicates}")
        print(f"  - Duplicate Ratio %: {dup_ratio:.1f}%")
        print(f"Discovery Metrics:")
        print(f"  - Discovery Rate %: {discovery_rate:.1f}% (Expected: 90%+)")
        print(f"Popularity Bias Audit:")
        print(f"  - Popularity Distribution: {pop_dist}")
        print(f"Seed Franchises checked: {list(seed_franchises)}")

if __name__ == "__main__":
    main()
