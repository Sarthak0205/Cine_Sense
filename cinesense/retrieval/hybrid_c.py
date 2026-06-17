import numpy as np


def hybrid_c_retrieval_scores(
    train_indices: np.ndarray,
    catalog_embeddings: np.ndarray,
    popularity_scores: np.ndarray,
    semantic_weight: float,
    popularity_weight: float,
    seed_batch_size: int = 128,
) -> np.ndarray:
    """Calculates candidate retrieval scores: semantic similarity + global popularity."""
    if catalog_embeddings is None or catalog_embeddings.size == 0:
        raise RuntimeError("Catalog embeddings are not initialized.")

    max_scores = np.full(len(catalog_embeddings), -np.inf, dtype=np.float32)
    batch_size = max(1, seed_batch_size)

    for start in range(0, train_indices.size, batch_size):
        batch_indices = train_indices[start : start + batch_size]
        train_embeddings = catalog_embeddings[batch_indices]
        batch_scores = catalog_embeddings @ train_embeddings.T
        max_scores = np.maximum(max_scores, batch_scores.max(axis=1))

    return semantic_weight * max_scores + popularity_weight * popularity_scores


def top_retrieval_indices(
    retrieval_scores: np.ndarray,
    train_items: set[int],
    anime_ids: np.ndarray,
    candidate_count: int,
) -> np.ndarray:
    """Selects top candidate_count item indices excluding already interacted items."""
    ranked_indices = np.argsort(-retrieval_scores, kind="mergesort")
    excluded_items = set(train_items)
    retrieved_indices = []

    for index in ranked_indices:
        anime_id = int(anime_ids[index])
        if anime_id in excluded_items:
            continue

        retrieved_indices.append(int(index))
        if len(retrieved_indices) == candidate_count:
            break

    return np.asarray(retrieved_indices, dtype=np.int32)
