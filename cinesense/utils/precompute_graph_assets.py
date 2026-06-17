import os
import sys
import json
import re
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import networkx as nx

from pathlib import Path
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def clean_title(t):
    return re.sub(r'[^a-z0-9]', '', str(t).lower())

def main():
    print("Loading catalog and user watches...", flush=True)
    
    # 1. Load twostage_v1 baseline assets to match IDs
    baseline_assets = np.load("cinesense/models/twostage_v1/model_assets.npz")
    anime_ids_v1 = baseline_assets["anime_ids"].astype(np.int32)
    id_to_v1_idx = {int(aid): idx for idx, aid in enumerate(anime_ids_v1.tolist())}
    
    # 2. Load user watches
    chunks = []
    for chunk in pd.read_csv("archive-2/user_watches.csv", usecols=["user_id", "anime_id", "score"], chunksize=5_000_000):
        pos_chunk = chunk[(chunk["score"] >= 7) & chunk["anime_id"].isin(id_to_v1_idx.keys())]
        chunks.append(pos_chunk)
    df_pos = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["user_id", "anime_id"])
    
    anime_counts = df_pos["anime_id"].value_counts()
    supported_animes = set(anime_counts[anime_counts >= 100].index)
    median_watches = anime_counts[anime_counts >= 100].median()
    supported_animes_list = sorted(list(supported_animes))
    anime_to_idx = {aid: idx for idx, aid in enumerate(supported_animes_list)}
    
    print(f"Number of supported anime (watches >= 100): {len(supported_animes_list)}", flush=True)
    print(f"Median watches of supported anime: {median_watches}", flush=True)

    # 3. Compute co-watches using sparse matrix multiplication
    print("Computing co-watches and Jaccard overlaps...", flush=True)
    df_pos_filtered = df_pos[df_pos["anime_id"].isin(supported_animes)].copy()
    user_unique = df_pos_filtered["user_id"].unique()
    user_to_idx = {uid: idx for idx, uid in enumerate(user_unique)}
    df_pos_filtered["user_idx"] = df_pos_filtered["user_id"].map(user_to_idx)
    df_pos_filtered["anime_idx"] = df_pos_filtered["anime_id"].map(anime_to_idx)
    
    X = csr_matrix((np.ones(len(df_pos_filtered), dtype=np.float32), 
                    (df_pos_filtered["user_idx"].values, df_pos_filtered["anime_idx"].values)), 
                   shape=(len(user_unique), len(supported_animes_list)))
    
    col_sums = np.array(X.sum(axis=0)).flatten()
    I = X.T.dot(X)
    I.data[I.data < 15] = 0
    I.eliminate_zeros()
    
    # Build Jaccard neighbors
    print("Filtering neighbors and building collaborative graph...", flush=True)
    I_coo = I.tocoo()
    r_idx = I_coo.row
    c_idx = I_coo.col
    intersect = I_coo.data
    unions = col_sums[r_idx] + col_sums[c_idx] - intersect
    jaccard_scores = intersect / unions

    # Threshold rules
    niche_mask = np.zeros(len(supported_animes_list), dtype=bool)
    for aid, idx in anime_to_idx.items():
        if anime_counts.get(aid, 0) < median_watches:
            niche_mask[idx] = True
            
    is_niche_pair = niche_mask[r_idx] | niche_mask[c_idx]
    pair_thresholds = np.where(is_niche_pair, 0.14, 0.18)
    threshold_mask = jaccard_scores >= pair_thresholds
    
    matching_rows = r_idx[threshold_mask]
    matching_cols = c_idx[threshold_mask]
    matching_jaccards = jaccard_scores[threshold_mask]
    
    # Organize neighbors using raw Jaccard scores
    neighbors_raw = {aid: [] for aid in supported_animes_list}
    for r, c, j in zip(r_idx, c_idx, jaccard_scores):
        r_aid = supported_animes_list[r]
        c_aid = supported_animes_list[c]
        if r_aid != c_aid: # No self-loops in neighbors
            neighbors_raw[r_aid].append((c_aid, float(j)))
            
    # Sort neighbors by Jaccard descending and keep top 200, then sort those 200 by ID for binary search
    num_nodes = len(supported_animes_list)
    neighbor_ids = np.full((num_nodes, 200), 2147483647, dtype=np.int32)
    neighbor_jaccards = np.zeros((num_nodes, 200), dtype=np.float32)
    
    for idx, aid in enumerate(supported_animes_list):
        candidates = neighbors_raw[aid]
        # Keep top 200
        candidates.sort(key=lambda x: -x[1])
        top_candidates = candidates[:200]
        # Sort by ID for binary search lookups
        top_candidates.sort(key=lambda x: x[0])
        
        for k_idx, (neighbor_id, jaccard_val) in enumerate(top_candidates):
            neighbor_ids[idx, k_idx] = neighbor_id
            neighbor_jaccards[idx, k_idx] = jaccard_val

    # 4. Construct networkx graph and compute distance-2 lookup matrix
    print("Constructing networkx graph and precomputing distance lookup...", flush=True)
    G = nx.Graph()
    G.add_nodes_from(supported_animes_list)
    for r, c in zip(matching_rows, matching_cols):
        r_aid = supported_animes_list[r]
        c_aid = supported_animes_list[c]
        if r_aid != c_aid:
            G.add_edge(r_aid, c_aid)
            
    distance_lookup = np.zeros((num_nodes, num_nodes), dtype=np.int8)
    for idx, aid in enumerate(supported_animes_list):
        paths = nx.single_source_shortest_path_length(G, aid, cutoff=2)
        for target_aid, dist in paths.items():
            if dist > 0:
                t_idx = anime_to_idx[target_aid]
                distance_lookup[idx, t_idx] = dist
                
    # 5. Output assets and run validation assertions
    output_dir = "cinesense/models/twostage_v1"
    output_path = os.path.join(output_dir, "graph_assets.npz")
    print(f"Saving assets to {output_path}...", flush=True)
    
    from datetime import datetime, UTC
    np.savez_compressed(
        output_path,
        neighbor_ids=neighbor_ids,
        neighbor_jaccards=neighbor_jaccards,
        distance_lookup=distance_lookup,
        supported_anime_ids=np.array(supported_animes_list, dtype=np.int32),
        col_sums=col_sums,
        graph_version=np.array("v1"),
        supported_items=np.array(len(supported_animes_list)),
        build_date=np.array(datetime.now(UTC).isoformat().replace("+00:00", "Z"))
    )
    
    # Assertions / Validations
    print("Running asset validations...", flush=True)
    asset_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Asset size: {asset_size_mb:.2f} MB")
    
    assert asset_size_mb < 250.0, f"Error: Asset size {asset_size_mb:.2f} MB exceeds 250 MB limit!"
    assert neighbor_ids.shape[1] == 200, "Error: Neighbor limit must be exactly 200"
    assert len(supported_animes_list) == len(col_sums), "Error: supported_animes and col_sums size mismatch"
    
    # Verify no duplicate edges in graph G
    assert nx.number_of_selfloops(G) == 0, "Error: Self loops detected in the collaborative graph!"
    print("Graph assets generated and verified successfully!", flush=True)

if __name__ == "__main__":
    main()
