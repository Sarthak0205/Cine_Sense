import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Set PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cinesense.utils.model_storage import load_model
from cinesense.services.recommendation import RecommendationService, get_franchise

# Detailed metadata for our target seeds and recommendations
anime_meta = {
    1535: {"genres": {"Supernatural", "Thriller", "Psychological", "Mystery"}, "studio": "Madhouse", "source": "Manga", "demographic": "Shounen"},
    19: {"genres": {"Thriller", "Psychological", "Mystery", "Drama"}, "studio": "Madhouse", "source": "Manga", "demographic": "Seinen"},
    13601: {"genres": {"Sci-Fi", "Action", "Psychological", "Suspense"}, "studio": "Production I.G", "source": "Original", "demographic": "None"},
    1575: {"genres": {"Sci-Fi", "Action", "Mecha", "Military", "Drama"}, "studio": "Sunrise", "source": "Original", "demographic": "None"},
    31043: {"genres": {"Thriller", "Supernatural", "Mystery", "Psychological"}, "studio": "A-1 Pictures", "source": "Manga", "demographic": "Seinen"},
    22535: {"genres": {"Sci-Fi", "Action", "Horror", "Gore"}, "studio": "Madhouse", "source": "Manga", "demographic": "Seinen"},
    13125: {"genres": {"Sci-Fi", "Mystery", "Psychological", "Drama"}, "studio": "A-1 Pictures", "source": "Novel", "demographic": "None"},
    9253: {"genres": {"Sci-Fi", "Thriller", "Suspense"}, "studio": "White Fox", "source": "Visual Novel", "demographic": "None"},
    30: {"genres": {"Sci-Fi", "Mecha", "Psychological", "Drama"}, "studio": "Gainax", "source": "Original", "demographic": "None"},
    10620: {"genres": {"Supernatural", "Thriller", "Psychological", "Action"}, "studio": "Asread", "source": "Manga", "demographic": "Shounen"},
    
    11061: {"genres": {"Action", "Adventure", "Fantasy", "Supernatural"}, "studio": "Madhouse", "source": "Manga", "demographic": "Shounen"},
    5114: {"genres": {"Action", "Adventure", "Fantasy", "Drama"}, "studio": "Bones", "source": "Manga", "demographic": "Shounen"},
    392: {"genres": {"Action", "Supernatural", "Martial Arts", "Comedy"}, "studio": "Pierrot", "source": "Manga", "demographic": "Shounen"},
    20: {"genres": {"Action", "Adventure", "Fantasy", "Martial Arts"}, "studio": "Pierrot", "source": "Manga", "demographic": "Shounen"},
    21: {"genres": {"Action", "Adventure", "Fantasy", "Comedy"}, "studio": "Toei Animation", "source": "Manga", "demographic": "Shounen"},
    31964: {"genres": {"Action", "Supernatural", "School", "Comedy"}, "studio": "Bones", "source": "Manga", "demographic": "Shounen"},
    269: {"genres": {"Action", "Supernatural", "Fantasy", "Martial Arts"}, "studio": "Pierrot", "source": "Manga", "demographic": "Shounen"},
    16498: {"genres": {"Action", "Drama", "Fantasy", "Military"}, "studio": "Wit Studio", "source": "Manga", "demographic": "Shounen"},
    32182: {"genres": {"Action", "Comedy", "Supernatural", "School"}, "studio": "Bones", "source": "Manga", "demographic": "Shounen"},
    
    6702: {"genres": {"Action", "Adventure", "Fantasy", "Comedy"}, "studio": "A-1 Pictures", "source": "Manga", "demographic": "Shounen"},
    813: {"genres": {"Action", "Martial Arts", "Adventure", "Sci-Fi"}, "studio": "Toei Animation", "source": "Manga", "demographic": "Shounen"},
    918: {"genres": {"Action", "Comedy", "Parody", "Historical"}, "studio": "Sunrise", "source": "Manga", "demographic": "Shounen"},
    14513: {"genres": {"Action", "Adventure", "Fantasy", "Magic"}, "studio": "A-1 Pictures", "source": "Manga", "demographic": "Shounen"},
    
    34572: {"genres": {"Action", "Fantasy", "Magic", "Adventure"}, "studio": "Pierrot", "source": "Manga", "demographic": "Shounen"},
    
    10087: {"genres": {"Action", "Supernatural", "Fantasy", "Drama"}, "studio": "ufotable", "source": "Light Novel", "demographic": "None"},
    22297: {"genres": {"Action", "Supernatural", "Fantasy", "Drama"}, "studio": "ufotable", "source": "Visual Novel", "demographic": "None"},
    2593: {"genres": {"Thriller", "Supernatural", "Mystery", "Action"}, "studio": "ufotable", "source": "Light Novel", "demographic": "None"},
    
    889: {"genres": {"Action", "Crime", "Thriller", "Drama"}, "studio": "Madhouse", "source": "Manga", "demographic": "Seinen"},
    12413: {"genres": {"Action", "Crime", "Drama", "Adventure"}, "studio": "White Fox", "source": "Manga", "demographic": "Seinen"},
    25183: {"genres": {"Action", "Crime", "Drama", "Supernatural"}, "studio": "Manglobe", "source": "Manga", "demographic": "Seinen"},
    6024: {"genres": {"Action", "Sci-Fi", "Drama", "Suspense"}, "studio": "P.A. Works", "source": "Original", "demographic": "None"},
    5682: {"genres": {"Action", "Thriller", "Drama", "Mystery"}, "studio": "Bee Train", "source": "Visual Novel", "demographic": "None"},
    1: {"genres": {"Sci-Fi", "Action", "Space", "Comedy"}, "studio": "Sunrise", "source": "Original", "demographic": "None"},
    4090: {"genres": {"Action", "Adventure", "Drama", "Comedy"}, "studio": "Manglobe", "source": "Original", "demographic": "None"},
    
    457: {"genres": {"Slice of Life", "Supernatural", "Mystery", "Adventure"}, "studio": "Artland", "source": "Manga", "demographic": "Seinen"},
    3002: {"genres": {"Slice of Life", "Supernatural", "Drama", "Fantasy"}, "studio": "Brain's Base", "source": "Manga", "demographic": "Shoujo"},
    482: {"genres": {"Adventure", "Slice of Life", "Drama", "Sci-Fi"}, "studio": "A.C.G.T", "source": "Light Novel", "demographic": "None"},
    2264: {"genres": {"Supernatural", "Mystery", "Horror", "Historical"}, "studio": "Toei Animation", "source": "Original", "demographic": "None"},
    22789: {"genres": {"Slice of Life", "Comedy", "Drama", "School"}, "studio": "Kinema Citrus", "source": "Manga", "demographic": "Shounen"},
    10408: {"genres": {"Supernatural", "Drama", "Romance", "Shojo"}, "studio": "Brain's Base", "source": "Manga", "demographic": "Shoujo"},
    33352: {"genres": {"Slice of Life", "Drama", "Fantasy", "Sci-Fi"}, "studio": "Kyoto Animation", "source": "Light Novel", "demographic": "None"}
}

title_to_id = {
    # Death Note recs
    "monster": 19,
    "psycho-pass": 13601,
    "code geass: lelouch of the rebellion": 1575,
    "erased": 31043,
    "parasyte -the maxim-": 22535,
    "shinsekai yori": 13125,
    "steins;gate": 9253,
    "neon genesis evangelion": 30,
    "mirai nikki": 10620,
    # HxH recs
    "fullmetal alchemist: brotherhood": 5114,
    "yu yu hakusho": 392,
    "naruto": 20,
    "one piece": 21,
    "my hero academia": 31964,
    "bleach": 269,
    "attack on titan": 16498,
    "mob psycho 100": 32182,
    # One Piece recs
    "fairy tail": 6702,
    "dragon ball z": 813,
    "gintama": 918,
    "magi: the labyrinth of magic": 14513,
    # Naruto recs
    "black clover": 34572,
    # Fate/Zero recs
    "fate/stay night: unlimited blade works": 22297,
    "the garden of sinners": 2593, # mapped to movie 1
    # Black Lagoon recs
    "jormungand": 12413,
    "gangsta.": 25183,
    "canaan": 6024,
    "phantom: requiem for the phantom": 5682,
    "cowboy bebop": 1,
    "michiko & hitchin": 4090,
    "fate/zero": 10087,
    # Mushishi recs
    "natsume's book of friends": 3002,
    "kino's journey": 482,
    "mononoke": 2264,
    "barakamon": 22789,
    "hotarubi no mori e": 10408,
    "violet evergarden": 33352,
    "hunter x hunter": 11061,
}

def get_overall_class(metadata_score, synopsis_score, col_jaccard):
    # Overall classification rules
    if col_jaccard >= 0.10 or metadata_score >= 0.40 or (col_jaccard >= 0.05 and metadata_score >= 0.20):
        return "Strong"
    elif col_jaccard >= 0.04 or metadata_score >= 0.20 or (col_jaccard >= 0.01 and metadata_score >= 0.10):
        return "Moderate"
    elif col_jaccard >= 0.005 or metadata_score >= 0.05 or synopsis_score >= 0.15:
        return "Weak"
    else:
        return "None"

def get_signal_class_col(jaccard):
    if jaccard >= 0.10: return "Strong"
    elif jaccard >= 0.04: return "Moderate"
    elif jaccard >= 0.005: return "Weak"
    else: return "None"

def get_signal_class_meta(score):
    if score >= 0.40: return "Strong"
    elif score >= 0.20: return "Moderate"
    elif score >= 0.05: return "Weak"
    else: return "None"

def get_signal_class_syn(score):
    if score >= 0.35: return "Strong"
    elif score >= 0.20: return "Moderate"
    elif score >= 0.08: return "Weak"
    else: return "None"

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

    # Target seeds for retraining audit
    target_seed_names = [
        "Hunter x Hunter",
        "One Piece",
        "Naruto",
        "Fullmetal Alchemist: Brotherhood",
        "Death Note",
        "Mushishi",
        "Black Lagoon",
        "Fate/Zero"
    ]

    target_entries = []
    for name in target_seed_names:
        for entry in gold_dataset:
            if entry["seed"].lower().strip() == name.lower().strip() or \
               (name == "Fullmetal Alchemist: Brotherhood" and entry["seed"] == "FMAB"):
                target_entries.append(entry)
                break

    # Get active target ID set
    all_target_ids = set()
    missing_pairs = [] # tuples: (seed_id, seed_name, rec_id, rec_title)

    for entry in target_entries:
        seed_name = entry["seed"]
        seed_id = entry["anime_id"]
        all_target_ids.add(seed_id)
        
        # Get baseline recommendations to identify missing ones
        recs = service.recommend([seed_id], ratings={seed_id: 10.0}, top_k=10, mode="discover")
        rec_ids = {r["anime_id"] for r in recs}
        
        # All target recommendations
        all_recs = entry["good_recommendations"] + entry["acceptable_recommendations"]
        for title in all_recs:
            rec_id = title_to_id.get(title.lower().strip())
            if rec_id is None:
                # Fallback to search
                for aid, meta in service.catalog_meta.items():
                    if meta["title"].lower().strip() == title.lower().strip():
                        rec_id = aid
                        break
            if rec_id is not None:
                all_target_ids.add(rec_id)
                # If missing from Top 10 recommendations
                if rec_id not in rec_ids:
                    missing_pairs.append((seed_id, seed_name, rec_id, title))
            else:
                print(f"Warning: Could not resolve title '{title}'")

    print(f"Total missing pairs to audit: {len(missing_pairs)}")
    print("Loading user watches to extract collaborative signals...", flush=True)
    t_start = time.time()
    chunks = []
    # Filter on the fly for our target set to be memory and time efficient
    for chunk in pd.read_csv("archive-2/user_watches.csv", chunksize=500000, usecols=["user_id", "anime_id", "score"]):
        filtered = chunk[chunk["anime_id"].isin(all_target_ids)]
        chunks.append(filtered)
    df_watches = pd.concat(chunks, ignore_index=True)
    print(f"Loaded {len(df_watches)} rows in {time.time() - t_start:.2f} seconds.")

    # Get positive user sets per anime (score >= 7)
    print("Building positive user sets...", flush=True)
    user_sets = {}
    for anime_id in all_target_ids:
        user_sets[anime_id] = set(df_watches[(df_watches["anime_id"] == anime_id) & (df_watches["score"] >= 7)]["user_id"])

    # Compute TF-IDF of synopses
    print("Computing TF-IDF of synopses...", flush=True)
    vectorizer = TfidfVectorizer(stop_words='english')
    # Use empty string if synopsis is missing
    synopses_list = [catalog_df.loc[i, 'synopsis'] if pd.notna(catalog_df.loc[i, 'synopsis']) else "" for i in range(len(catalog_df))]
    tfidf_matrix = vectorizer.fit_transform(synopses_list)

    # Run audit per pair
    rows = []
    
    for seed_id, seed_name, rec_id, rec_title in missing_pairs:
        # 1. Metadata overlap
        meta_s = anime_meta.get(seed_id, {"genres": set(), "studio": "None", "source": "None", "demographic": "None"})
        meta_r = anime_meta.get(rec_id, {"genres": set(), "studio": "None", "source": "None", "demographic": "None"})
        
        g_s = meta_s["genres"]
        g_r = meta_r["genres"]
        jaccard_genres = len(g_s & g_r) / len(g_s | g_r) if len(g_s | g_r) > 0 else 0.0
        
        studio_match = 1.0 if meta_s["studio"] == meta_r["studio"] else 0.0
        source_match = 1.0 if meta_s["source"] == meta_r["source"] else 0.0
        demo_match = 1.0 if meta_s["demographic"] == meta_r["demographic"] and meta_s["demographic"] != "None" else 0.0
        
        metadata_score = 0.5 * jaccard_genres + 0.2 * studio_match + 0.15 * source_match + 0.15 * demo_match
        
        # 2. Synopsis similarity
        idx_s = model.item_id_to_index[seed_id]
        idx_r = model.item_id_to_index[rec_id]
        
        emb_s = model.catalog_embeddings[idx_s]
        emb_r = model.catalog_embeddings[idx_r]
        emb_sim = float(np.dot(emb_s, emb_r))
        
        tfidf_sim = float((tfidf_matrix[idx_s] @ tfidf_matrix[idx_r].T).toarray()[0, 0])
        synopsis_score = 0.5 * emb_sim + 0.5 * tfidf_sim
        
        # 3. Collaborative signals
        u_s = user_sets.get(seed_id, set())
        u_r = user_sets.get(rec_id, set())
        
        col_jaccard = len(u_s & u_r) / len(u_s | u_r) if len(u_s | u_r) > 0 else 0.0
        
        # Classifications
        m_class = get_signal_class_meta(metadata_score)
        s_class = get_signal_class_syn(synopsis_score)
        c_class = get_signal_class_col(col_jaccard)
        overall_class = get_overall_class(metadata_score, synopsis_score, col_jaccard)
        
        rows.append({
            "seed": seed_name,
            "target": rec_title,
            "pair": f"{seed_name} -> {rec_title}",
            "meta_val": metadata_score,
            "meta_class": m_class,
            "syn_val": synopsis_score,
            "syn_class": s_class,
            "col_val": col_jaccard,
            "col_class": c_class,
            "overall": overall_class
        })

    df_results = pd.DataFrame(rows)
    
    # Sort results by overall signal quality
    overall_order = {"Strong": 0, "Moderate": 1, "Weak": 2, "None": 3}
    df_results["overall_rank"] = df_results["overall"].map(overall_order)
    df_results = df_results.sort_values(by="overall_rank").reset_index(drop=True)

    print("\n" + "="*80)
    print("EMBEDDING RETRAINING READINESS SIGNAL COVERAGE")
    print("="*80)
    print(f"| {'Missing Relationship':<42} | {'Metadata':<10} | {'Synopsis':<10} | {'Collaborative':<13} | {'Overall Signal':<14} |")
    print(f"| {'-'*42} | {'-'*10} | {'-'*10} | {'-'*13} | {'-'*14} |")
    for _, r in df_results.iterrows():
        print(f"| {r['pair']:<42} | {r['meta_class']:<10} | {r['syn_class']:<10} | {r['col_class']:<13} | {r['overall']:<14} |")
    print()

    # Counts
    total = len(df_results)
    strong_cnt = sum(df_results["overall"] == "Strong")
    mod_cnt = sum(df_results["overall"] == "Moderate")
    weak_cnt = sum(df_results["overall"] == "Weak")
    none_cnt = sum(df_results["overall"] == "None")
    
    strong_pct = (strong_cnt / total) * 100.0
    mod_pct = (mod_cnt / total) * 100.0
    weak_pct = (weak_cnt / total) * 100.0
    none_pct = (none_cnt / total) * 100.0
    
    print("="*80)
    print("GLOBAL SIGNAL AUDIT RESULTS SUMMARY")
    print("="*80)
    print(f"Total Missing Recommendations Evaluated: {total}")
    print(f"  - Strong overall signals:    {strong_cnt} ({strong_pct:.2f}%)")
    print(f"  - Moderate overall signals:  {mod_cnt} ({mod_pct:.2f}%)")
    print(f"  - Weak overall signals:      {weak_cnt} ({weak_pct:.2f}%)")
    print(f"  - No training signals:       {none_cnt} ({none_pct:.2f}%)")
    print()
    
    print("Decision Gate Metrics:")
    print(f"  - Strong + Moderate signals: {strong_pct + mod_pct:.2f}%")
    print(f"  - Weak + No signals:         {weak_pct + none_pct:.2f}%")
    print()
    
if __name__ == "__main__":
    main()
