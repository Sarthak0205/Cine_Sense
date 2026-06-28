import os
import sys
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.audit_common import (
    load_model_c,
    load_model_d,
    write_markdown_table,
    save_report,
)
from cinesense.utils.franchise import get_canonical_franchise
from cinesense.services.recommendation import is_sequel_title

def run_retrieval_stage(service, seed_ids, k_retrieval=100):
    recommender = service.recommender
    train_indices = np.asarray([recommender.item_id_to_index[aid] for aid in seed_ids if aid in recommender.item_id_to_index], dtype=np.int32)
    if train_indices.size == 0:
        return []
    
    from cinesense.retrieval.hybrid_c import hybrid_c_retrieval_scores, top_retrieval_indices
    retrieval_scores = hybrid_c_retrieval_scores(
        train_indices,
        recommender.catalog_embeddings,
        recommender.popularity_scores,
        recommender.semantic_weight,
        recommender.popularity_weight,
        recommender.seed_batch_size,
    )
    raw_indices = top_retrieval_indices(
        retrieval_scores,
        set(seed_ids),
        recommender.anime_ids,
        k_retrieval,
    )
    return [int(recommender.anime_ids[idx]) for idx in raw_indices]

def evaluate_leakage(service, scenario_name, seed_ids):
    # Run recommendations at top_k = 50 to compute Leakage@10, @20, @50
    recs = service.recommend(anime_ids=seed_ids, top_k=50, mode="discover", user_id="leakage_audit")
    rec_ids = [r["anime_id"] for r in recs]
    rec_titles = [r["title"] for r in recs]
    
    # Retrieval candidates
    retrieved_ids = run_retrieval_stage(service, seed_ids, k_retrieval=100)
    
    # Get seed franchises
    seed_franchises = set()
    for sid in seed_ids:
        meta = service.catalog_meta.get(sid, {})
        title = meta.get("title", "")
        title_eng = meta.get("title_english", "")
        seed_franchises.add(get_canonical_franchise(title))
        if title_eng:
            seed_franchises.add(get_canonical_franchise(title_eng))
    seed_franchises.discard("")
    
    # Compute leakage at different cutoffs
    leakage_10 = 0
    leakage_20 = 0
    leakage_50 = 0
    violations = []
    
    for i, rid in enumerate(rec_ids):
        meta = service.catalog_meta.get(rid, {})
        title = meta.get("title", "")
        title_eng = meta.get("title_english", "")
        f_title = get_canonical_franchise(title)
        f_eng = get_canonical_franchise(title_eng) if title_eng else ""
        
        is_leaked = (f_title in seed_franchises) or (f_eng and f_eng in seed_franchises)
        if is_leaked:
            if i < 10:
                leakage_10 += 1
            if i < 20:
                leakage_20 += 1
            if i < 50:
                leakage_50 += 1
            violations.append(f"Rank {i+1}: {title} (Seed Franchise Match)")
            
        # Sequel check: if recommended in discover mode, verify if it is a sequel
        # A recommended item in discover mode should not be a sequel to ANY franchise
        root_id = service.get_franchise_root(f_title)
        is_seq = False
        if root_id is not None and rid != root_id:
            is_seq = True
        elif service.is_sequel_title(title) or (title_eng and service.is_sequel_title(title_eng)):
            is_seq = True
            
        if is_seq:
            if i < 10:
                violations.append(f"Rank {i+1}: {title} (Sequel title in discover mode)")
    
    # Duplicate franchise count in top-10
    franchises_seen = set()
    duplicates_count = 0
    for rid in rec_ids[:10]:
        meta = service.catalog_meta.get(rid, {})
        f_title = get_canonical_franchise(meta.get("title", ""))
        if f_title in franchises_seen:
            duplicates_count += 1
            violations.append(f"Rank {rec_ids.index(rid)+1}: {meta.get('title', '')} (Duplicate Franchise: {f_title})")
        else:
            franchises_seen.add(f_title)
            
    # Retrieval leakage (fraction of top 100 retrieval candidates that belong to seed franchise)
    ret_leakage_count = 0
    for rid in retrieved_ids:
        meta = service.catalog_meta.get(rid, {})
        f_title = get_canonical_franchise(meta.get("title", ""))
        f_eng = get_canonical_franchise(meta.get("title_english", "")) if meta.get("title_english") else ""
        if (f_title in seed_franchises) or (f_eng and f_eng in seed_franchises):
            ret_leakage_count += 1
            
    ret_leakage_pct = (ret_leakage_count / len(retrieved_ids)) * 100 if retrieved_ids else 0.0
    
    return {
        "leakage_10": (leakage_10 / 10.0) * 100,
        "leakage_20": (leakage_20 / 20.0) * 100,
        "leakage_50": (leakage_50 / len(rec_ids)) * 100 if rec_ids else 0.0,
        "duplicates": duplicates_count,
        "retrieval_leakage": ret_leakage_pct,
        "final_leakage": (leakage_10 / 10.0) * 100, # Final leakage is leakage@10
        "violations": violations
    }

def main():
    print("Running Franchise Leakage Stress Test...", flush=True)
    
    service_c = load_model_c()
    service_d = load_model_d()
    
    scenarios = [
        # Single Seed
        {"name": "Naruto (Single)", "seeds": [20]},
        {"name": "One Piece (Single)", "seeds": [21]},
        {"name": "Bleach (Single)", "seeds": [269]},
        {"name": "Attack on Titan (Single)", "seeds": [16498]},
        {"name": "Death Note (Single)", "seeds": [1535]},
        {"name": "Code Geass (Single)", "seeds": [1575]},
        {"name": "Steins;Gate (Single)", "seeds": [9253]},
        {"name": "Clannad (Single)", "seeds": [2167]},
        {"name": "Violet Evergarden (Single)", "seeds": [33352]},
        {"name": "Fate/Zero (Single)", "seeds": [10087]},
        
        # Multi Seed
        {"name": "Naruto + One Piece", "seeds": [20, 21]},
        {"name": "Death Note + Code Geass", "seeds": [1535, 1575]},
        {"name": "Steins;Gate + Code Geass", "seeds": [9253, 1575]},
        
        # Sequel as Seed
        {"name": "Naruto Shippuden (Sequel Seed)", "seeds": [1735]},
        {"name": "Clannad: After Story (Sequel Seed)", "seeds": [4181]},
        {"name": "Steins;Gate 0 (Sequel Seed)", "seeds": [30484]},
        {"name": "Code Geass R2 (Sequel Seed)", "seeds": [2904]},
    ]
    
    rows_c = []
    rows_d = []
    all_violations_c = []
    all_violations_d = []
    
    for sc in scenarios:
        name = sc["name"]
        seeds = sc["seeds"]
        
        res_c = evaluate_leakage(service_c, name, seeds)
        res_d = evaluate_leakage(service_d, name, seeds)
        
        rows_c.append([
            name,
            f"{res_c['leakage_10']:.1f}%",
            f"{res_c['leakage_20']:.1f}%",
            f"{res_c['leakage_50']:.1f}%",
            res_c['duplicates'],
            f"{res_c['retrieval_leakage']:.1f}%",
            f"{res_c['final_leakage']:.1f}%"
        ])
        if res_c["violations"]:
            all_violations_c.append((name, res_c["violations"]))
            
        rows_d.append([
            name,
            f"{res_d['leakage_10']:.1f}%",
            f"{res_d['leakage_20']:.1f}%",
            f"{res_d['leakage_50']:.1f}%",
            res_d['duplicates'],
            f"{res_d['retrieval_leakage']:.1f}%",
            f"{res_d['final_leakage']:.1f}%"
        ])
        if res_d["violations"]:
            all_violations_d.append((name, res_d["violations"]))

    # Pass/Fail evaluation
    # Success Criteria: Leakage@10 = 0%, Leakage@20 = 0%, Duplicate Franchise Count = 0
    def check_pass(rows):
        for r in rows:
            l10 = float(r[1].replace("%", ""))
            l20 = float(r[2].replace("%", ""))
            dups = int(r[4])
            if l10 > 0.0 or l20 > 0.0 or dups > 0:
                return "FAIL"
        return "PASS"
        
    pass_c = check_pass(rows_c)
    pass_d = check_pass(rows_d)
    
    # Build report content
    report_content = []
    report_content.append("# CineSense Franchise Leakage Audit Report")
    report_content.append(f"\n## Audit Summary")
    report_content.append(f"* **Model C Status**: **{pass_c}**")
    report_content.append(f"* **Model D Status**: **{pass_d}**")
    
    headers = ["Scenario", "Leakage@10", "Leakage@20", "Leakage@50", "Duplicate Franchises", "Retrieval Leakage", "Final Leakage"]
    
    report_content.append(f"\n## Model C (Locked Baseline) Results")
    report_content.append(write_markdown_table(headers, rows_c))
    
    report_content.append(f"\n## Model D (Production Candidate) Results")
    report_content.append(write_markdown_table(headers, rows_d))
    
    report_content.append(f"\n## Offending Titles & Violations")
    
    report_content.append(f"\n### Model C Violations")
    if all_violations_c:
        for name, viols in all_violations_c:
            report_content.append(f"\n* **Scenario: {name}**")
            for v in viols:
                report_content.append(f"  - {v}")
    else:
        report_content.append("None detected.")
        
    report_content.append(f"\n### Model D Violations")
    if all_violations_d:
        for name, viols in all_violations_d:
            report_content.append(f"\n* **Scenario: {name}**")
            for v in viols:
                report_content.append(f"  - {v}")
    else:
        report_content.append("None detected.")
        
    save_report("franchise_leakage_audit.md", "\n".join(report_content))

if __name__ == "__main__":
    main()
