import json
import os
from datetime import datetime, UTC
import numpy as np
import pandas as pd
from cinesense.recommenders.two_stage import CineSenseTwoStage


def save_model(
    model: CineSenseTwoStage,
    catalog_df: pd.DataFrame,
    dir_path: str,
    model_version: str,
    catalog_version: str,
    embedding_version: str,
) -> None:
    """Serializes model parameters and catalog metadata into production assets."""
    os.makedirs(dir_path, exist_ok=True)

    # 1. Save metadata
    metadata = {
        "model_version": model_version,
        "catalog_version": catalog_version,
        "embedding_version": embedding_version,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "hyperparameters": {
            "semantic_weight": model.semantic_weight,
            "popularity_weight": model.popularity_weight,
            "rating_weight_scheme": model.rating_weight_scheme,
            "retrieval_candidate_count": model.retrieval_candidate_count,
            "seed_batch_size": getattr(model, "seed_batch_size", 128),
        },
    }
    with open(os.path.join(dir_path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 2. Save numpy assets
    np.savez_compressed(
        os.path.join(dir_path, "model_assets.npz"),
        catalog_embeddings=model.catalog_embeddings,
        popularity_scores=model.popularity_scores,
        anime_ids=model.anime_ids,
    )

    # 3. Save catalog Parquet
    # We only save the metadata columns required at inference (e.g. titles, synopsis, etc.)
    # matching the original catalog's structure
    catalog_df.to_parquet(
        os.path.join(dir_path, "catalog.parquet"),
        index=False,
    )


def load_model(dir_path: str) -> tuple[CineSenseTwoStage, pd.DataFrame, dict]:
    """Loads the serialized model assets and catalog metadata."""
    # 1. Load metadata
    with open(os.path.join(dir_path, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # 2. Load numpy assets
    assets = np.load(os.path.join(dir_path, "model_assets.npz"))
    catalog_embeddings = assets["catalog_embeddings"]
    popularity_scores = assets["popularity_scores"]
    anime_ids = assets["anime_ids"]

    # Reconstruct item_id_to_index
    item_id_to_index = {
        int(item_id): index for index, item_id in enumerate(anime_ids.tolist())
    }

    # 3. Load catalog
    catalog_df = pd.read_parquet(os.path.join(dir_path, "catalog.parquet"))

    # 4. Instantiate model
    hparams = metadata["hyperparameters"]
    model = CineSenseTwoStage(
        semantic_weight=hparams["semantic_weight"],
        popularity_weight=hparams["popularity_weight"],
        rating_weight_scheme=hparams["rating_weight_scheme"],
        retrieval_candidate_count=hparams["retrieval_candidate_count"],
    )
    model.seed_batch_size = hparams.get("seed_batch_size", 128)
    model.catalog_embeddings = catalog_embeddings
    model.popularity_scores = popularity_scores
    model.anime_ids = anime_ids
    model.item_id_to_index = item_id_to_index

    # Calculate popularity percentiles
    sorted_pop_indices = np.argsort(popularity_scores)
    pop_percentiles = np.zeros_like(popularity_scores)
    for rank, idx in enumerate(sorted_pop_indices):
        pop_percentiles[idx] = rank / len(popularity_scores)
    model.pop_percentiles = pop_percentiles

    # Load graph assets if present
    graph_path = os.path.join(dir_path, "graph_assets.npz")
    model.graph_available = False
    model.neighbor_ids = None
    model.neighbor_jaccards = None
    model.distance_lookup = None
    model.supported_anime_ids = None
    model.col_sums = None
    model.anime_to_idx = {}

    if os.path.exists(graph_path):
        try:
            print(f"Loading graph assets from: {graph_path}...", flush=True)
            graph_assets = np.load(graph_path)
            model.neighbor_ids = graph_assets["neighbor_ids"]
            model.neighbor_jaccards = graph_assets["neighbor_jaccards"]
            model.distance_lookup = graph_assets["distance_lookup"]
            model.supported_anime_ids = graph_assets["supported_anime_ids"]
            model.col_sums = graph_assets["col_sums"]

            # Map anime ID to index in the supported_anime_ids array
            model.anime_to_idx = {int(aid): idx for idx, aid in enumerate(model.supported_anime_ids.tolist())}

            # Graph Asset Validation checks:
            # 1. Version Validation
            expected_version = "v1"
            if "graph_version" not in graph_assets:
                raise ValueError("Validation failed: graph_version metadata missing from assets.")
            loaded_version = str(graph_assets["graph_version"].item())
            if loaded_version != expected_version:
                raise ValueError(f"Validation failed: expected version '{expected_version}', got '{loaded_version}'.")

            # 2. Shape Validation
            if model.neighbor_ids.shape != model.neighbor_jaccards.shape:
                raise ValueError(f"Validation failed: shape mismatch between neighbor_ids {model.neighbor_ids.shape} and neighbor_jaccards {model.neighbor_jaccards.shape}.")

            # 3. ID Uniqueness
            if len(set(model.supported_anime_ids)) != len(model.supported_anime_ids):
                raise ValueError("Validation failed: supported_anime_ids contains duplicate IDs.")

            # 4. Row Sorting (every row sorted ascending)
            if not np.all(model.neighbor_ids[:, :-1] <= model.neighbor_ids[:, 1:]):
                raise ValueError("Validation failed: neighbor_ids rows are not sorted ascending.")

            # 5. NaN Validation
            if model.neighbor_jaccards.dtype.kind in ('f', 'c'):
                if np.any(np.isnan(model.neighbor_jaccards)):
                    raise ValueError("Validation failed: neighbor_jaccards contains NaN values.")

            # 6. Distance Validation (non-zero distance values must be strictly in {1, 2})
            unique_distances = set(np.unique(model.distance_lookup)) - {0}
            if not unique_distances.issubset({1, 2}):
                raise ValueError(f"Validation failed: distance_lookup contains invalid non-zero distances {unique_distances}.")

            model.graph_available = True
            print("Graph assets loaded and validated successfully.", flush=True)
        except Exception as e:
            print(f"WARNING: Failed to load graph assets: {e}. Falling back to semantic-only.", flush=True)

    # Build startup-time immutable franchise index
    build_franchise_index(model, catalog_df)

    return model, catalog_df, metadata


def build_franchise_index(model: CineSenseTwoStage, catalog_df: pd.DataFrame) -> None:
    """Constructs anime_id_to_root, franchise_name_to_root, and root_to_members mappings on startup."""
    from cinesense.services.recommendation import get_franchise, is_sequel_title

    # 1. Group all anime IDs and their titles by franchise names
    franchise_to_candidates = {}
    for _, row in catalog_df.iterrows():
        aid = int(row["anime_id"])
        title = str(row.get("title", ""))
        title_eng = str(row.get("title_english", "")) if pd.notna(row.get("title_english")) else ""

        f_title = get_franchise(title)
        f_eng = get_franchise(title_eng) if title_eng else ""

        # We index candidates under both primary and English franchise names
        for f_name in {f_title, f_eng}:
            if not f_name:
                continue
            if f_name not in franchise_to_candidates:
                franchise_to_candidates[f_name] = []
            franchise_to_candidates[f_name].append((aid, title))

    # 2. Determine the franchise root for each franchise name
    anime_id_to_root = {}
    franchise_name_to_root = {}
    root_to_members = {}

    for f_name, candidates in franchise_to_candidates.items():
        # Find non-sequels
        non_sequels = []
        for aid, title in candidates:
            if not is_sequel_title(title):
                non_sequels.append((aid, title))

        # Select root ID
        if non_sequels:
            # Sort by title length first, then popularity descending
            non_sequels.sort(
                key=lambda x: (
                    len(x[1]),
                    -model.popularity_scores[model.item_id_to_index[x[0]]] if x[0] in model.item_id_to_index else 0.0
                )
            )
            root_id = non_sequels[0][0]
        else:
            # If all are sequels, pick the one with the shortest title
            candidates.sort(
                key=lambda x: (
                    len(x[1]),
                    -model.popularity_scores[model.item_id_to_index[x[0]]] if x[0] in model.item_id_to_index else 0.0
                )
            )
            root_id = candidates[0][0]

        franchise_name_to_root[f_name] = root_id

        if root_id not in root_to_members:
            root_to_members[root_id] = []

        for aid, _ in candidates:
            if aid not in root_to_members[root_id]:
                root_to_members[root_id].append(aid)

    # 3. Populate anime_id_to_root mapping
    for root_id, members in root_to_members.items():
        for aid in members:
            anime_id_to_root[aid] = root_id

    # Attach to the model
    model.anime_id_to_root = anime_id_to_root
    model.franchise_name_to_root = franchise_name_to_root
    model.root_to_members = root_to_members
