from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from evaluation.cinesense_single_seed_max import CineSenseSingleSeedMaxRecommender
from evaluation.datasets import ITEM_ID_COL


DEFAULT_MODEL_NAME = "cinesense_hybrid_lite"


@dataclass
class CineSenseHybridLite(CineSenseSingleSeedMaxRecommender):
    """Single-seed max semantic retrieval with a global train-popularity prior."""

    model_name: str = DEFAULT_MODEL_NAME
    semantic_weight: float = 0.90
    popularity_weight: float = 0.10
    popularity_scores: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32),
        init=False,
        repr=False,
    )

    def fit(
        self,
        anime_catalog: pd.DataFrame,
        train_interactions: pd.DataFrame,
        *_: Any,
        **__: Any,
    ) -> "CineSenseHybridLite":
        """Fit semantic embeddings and normalized train-only popularity scores."""

        super().fit(anime_catalog)
        self.popularity_scores = self._build_popularity_scores(train_interactions)
        return self

    def recommend(self, user_id: int, train_items: set[int], k: int) -> list[int]:
        """Recommend ranked anime IDs using semantic score plus global popularity prior."""

        if k <= 0:
            raise ValueError("k must be greater than 0.")
        if self.catalog_embeddings is None:
            raise RuntimeError("CineSenseHybridLite must be fitted before recommend().")
        if self.popularity_scores.size != len(self.anime_ids):
            raise RuntimeError("Popularity scores are not initialized correctly.")

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

        semantic_scores = self._max_similarity_to_train_items(train_indices)
        final_scores = (
            self.semantic_weight * semantic_scores
            + self.popularity_weight * self.popularity_scores
        )
        ranked_indices = np.argsort(-final_scores, kind="mergesort")

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

    def _build_popularity_scores(self, train_interactions: pd.DataFrame) -> np.ndarray:
        item_counts = train_interactions[ITEM_ID_COL].value_counts(sort=False)
        max_count = item_counts.max()
        popularity_scores = np.zeros(len(self.anime_ids), dtype=np.float32)

        if max_count == 0:
            return popularity_scores

        for item_id, count in item_counts.items():
            item_id = int(item_id)
            if item_id not in self.item_id_to_index:
                continue

            popularity_scores[self.item_id_to_index[item_id]] = float(count) / float(max_count)

        return popularity_scores
