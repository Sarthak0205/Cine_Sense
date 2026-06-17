from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evaluation.cinesense_recommender import CineSenseV1Recommender


DEFAULT_MODEL_NAME = "cinesense_single_seed_max"


@dataclass
class CineSenseSingleSeedMaxRecommender(CineSenseV1Recommender):
    """Semantic recommender that scores candidates by max similarity to any train item."""

    model_name: str = DEFAULT_MODEL_NAME
    seed_batch_size: int = 128

    def recommend(self, user_id: int, train_items: set[int], k: int) -> list[int]:
        """Recommend ranked anime IDs using max candidate-to-train-item similarity."""

        if k <= 0:
            raise ValueError("k must be greater than 0.")
        if self.catalog_embeddings is None:
            raise RuntimeError(
                "CineSenseSingleSeedMaxRecommender must be fitted before recommend()."
            )

        train_indices = np.asarray(
            [
                self.item_id_to_index[item_id]
                for item_id in train_items
                if item_id in self.item_id_to_index
            ],
            dtype=np.int32,
        )
        if train_indices.size == 0:
            return []

        similarity_scores = self._max_similarity_to_train_items(train_indices)
        ranked_indices = np.argsort(-similarity_scores, kind="mergesort")

        excluded_items = set(train_items)
        recommendations: list[int] = []
        seen_items: set[int] = set()

        for index in ranked_indices:
            anime_id = int(self.anime_ids[index])
            if anime_id in excluded_items or anime_id in seen_items:
                continue

            seen_items.add(anime_id)
            recommendations.append(anime_id)
            if len(recommendations) == k:
                break

        return recommendations

    def _max_similarity_to_train_items(self, train_indices: np.ndarray) -> np.ndarray:
        if self.catalog_embeddings is None:
            raise RuntimeError(
                "CineSenseSingleSeedMaxRecommender must be fitted before scoring."
            )

        max_scores = np.full(len(self.anime_ids), -np.inf, dtype=np.float32)
        batch_size = max(1, self.seed_batch_size)

        for start in range(0, train_indices.size, batch_size):
            batch_indices = train_indices[start : start + batch_size]
            train_embeddings = self.catalog_embeddings[batch_indices]
            batch_scores = self.catalog_embeddings @ train_embeddings.T
            max_scores = np.maximum(max_scores, batch_scores.max(axis=1))

        return max_scores
