import os
import sys
import re
import numpy as np
import pandas as pd

# Set path relative to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cinesense.utils.model_storage import load_model
from cinesense.services.recommendation import RecommendationService, get_franchise

def main():
    MODEL_DIR = os.path.join(PROJECT_ROOT, "cinesense/models/twostage_v1")
    model, catalog_df, metadata = load_model(MODEL_DIR)
    service = RecommendationService(model, catalog_df)

    seeds = [
        {"id": 1535, "name": "Death Note"},
        {"id": 1575, "name": "Code Geass"},
        {"id": 9253, "name": "Steins;Gate"},
        {"id": 5114, "name": "Fullmetal Alchemist Brotherhood"},
        {"id": 11061, "name": "Hunter x Hunter (2011)"},
        {"id": 16498, "name": "Attack on Titan"},
        {"id": 21, "name": "One Piece"},
        {"id": 19, "name": "Monster"},
    ]

    results = {}

    for seed in seeds:
        seed_id = seed["id"]
        seed_name = seed["name"]
        
        # Verify seed ID in index
        if seed_id not in model.item_id_to_index:
            print(f"Warning: Seed {seed_name} (ID: {seed_id}) not in model index!")
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
        
        # Sort indices (descending similarity)
        sorted_indices = np.argsort(-similarities)

        # Retrieve top 100 semantic neighbors (excluding the seed itself)
        neighbors = []
        for idx in sorted_indices:
            cand_id = int(model.anime_ids[idx])
            if cand_id == seed_id:
                continue
            neighbors.append((cand_id, float(similarities[idx])))
            if len(neighbors) == 100:
                break

        surviving = []
        removals = {
            "Same Franchise": 0,
            "Sequel": 0,
            "Movie": 0,
            "OVA": 0,
            "ONA": 0,
            "Special": 0,
            "Recap": 0,
        }

        for cand_id, sim in neighbors:
            meta = service.catalog_meta[cand_id]
            title = meta["title"]
            eng_title = meta.get("title_english") or ""

            t_low = title.lower()
            eng_low = eng_title.lower()

            # 1. Same Franchise check
            cand_f = get_franchise(title)
            cand_f_eng = get_franchise(eng_title) if eng_title else ""
            is_same_f = (cand_f in seed_franchises) or (cand_f_eng and cand_f_eng in seed_franchises)
            if is_same_f:
                removals["Same Franchise"] += 1
                continue

            # 2. Recap check
            if "recap" in t_low or "recap" in eng_low:
                removals["Recap"] += 1
                continue

            # 3. Movie check
            if "movie" in t_low or "movie" in eng_low or "film" in t_low or "film" in eng_low:
                removals["Movie"] += 1
                continue

            # 4. OVA check
            if "ova" in t_low or "ova" in eng_low:
                removals["OVA"] += 1
                continue

            # 5. ONA check
            if "ona" in t_low or "ona" in eng_low:
                removals["ONA"] += 1
                continue

            # 6. Special check
            if "special" in t_low or "special" in eng_low or "pilot" in t_low or "pilot" in eng_low:
                removals["Special"] += 1
                continue

            # 7. Sequel check
            root_id = service.get_franchise_root(cand_f)
            is_sequel = False
            if root_id is not None and cand_id != root_id:
                is_sequel = True
            elif service.is_sequel_title(title) or (eng_title and service.is_sequel_title(eng_title)):
                is_sequel = True

            if is_sequel:
                removals["Sequel"] += 1
                continue

            # Survives all filters
            surviving.append((cand_id, title, eng_title, sim))

        results[seed_name] = {
            "surviving": surviving,
            "removals": removals,
            "raw_count": len(neighbors),
            "survive_count": len(surviving),
        }

    # Print Report Output
    print("\n" + "="*80)
    print("AUDIT RESULTS: CINESENSE V1 FINAL RECOMMENDATION SYSTEM VALIDATION")
    print("="*80 + "\n")

    # Table A: Discovery Survival Analysis
    print("### Table A — Discovery Survival Analysis")
    print(f"| {'Seed':<32} | {'Raw Neighbors':<13} | {'Survive Filters':<15} | {'Survival %':<10} |")
    print(f"| {'-'*32} | {'-'*13} | {'-'*15} | {'-'*10} |")
    for seed_name, data in results.items():
        rate_str = f"{data['survive_count'] / data['raw_count']:.1%}"
        print(f"| {seed_name:<32} | {data['raw_count']:<13} | {data['survive_count']:<15} | {rate_str:<10} |")
    print()

    # Table B: Removal Breakdown
    print("### Table B — Removal Breakdown")
    print(f"| {'Seed':<32} | {'Same Franchise':<14} | {'Sequel':<6} | {'Movie':<5} | {'OVA':<3} | {'ONA':<3} | {'Special':<7} | {'Recap':<5} |")
    print(f"| {'-'*32} | {'-'*14} | {'-'*6} | {'-'*5} | {'-'*3} | {'-'*3} | {'-'*7} | {'-'*5} |")
    for seed_name, data in results.items():
        rem = data["removals"]
        print(f"| {seed_name:<32} | {rem['Same Franchise']:<14} | {rem['Sequel']:<6} | {rem['Movie']:<5} | {rem['OVA']:<3} | {rem['ONA']:<3} | {rem['Special']:<7} | {rem['Recap']:<5} |")
    print()

    # Table C: Top Discovery Candidates (Top 20 surviving)
    print("### Table C — Top Discovery Candidates")
    for seed_name, data in results.items():
        print(f"\n#### {seed_name}")
        surviving = data["surviving"]
        for rank, (cand_id, title, eng_title, sim) in enumerate(surviving[:20], 1):
            display_title = eng_title if eng_title else title
            print(f"{rank}. {display_title} (ID: {cand_id}, Similarity: {sim:.4f})")
    print()

    # Table D: Neighborhood Classification
    print("### Table D — Neighborhood Classification")
    print(f"| {'Seed':<32} | {'Survival %':<10} | {'Classification':<30} |")
    print(f"| {'-'*32} | {'-'*10} | {'-'*30} |")
    for seed_name, data in results.items():
        survival_rate = data["survive_count"] / data["raw_count"]
        if survival_rate >= 0.40:
            classification = "Rich Discovery Neighborhood"
        elif survival_rate >= 0.20:
            classification = "Moderate Discovery Neighborhood"
        else:
            classification = "Sparse Discovery Neighborhood"
        rate_str = f"{survival_rate:.1%}"
        print(f"| {seed_name:<32} | {rate_str:<10} | {classification:<30} |")
    print()

if __name__ == "__main__":
    main()
