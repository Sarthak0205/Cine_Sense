from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import pandas as pd

from cinesense.utils.text import (
    ITEM_ID_COL,
    TITLE_COL,
    ENGLISH_TITLE_COL,
    SYNOPSIS_COL,
    build_catalog_tags,
)

RatingWeightScheme = Literal["raw_score", "normalized", "strong"]
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@dataclass
class CineSenseV1Recommender:
    """Semantic CineSense recommender adapted to the benchmark interface."""

    model_name: str = "cinesense_v1"
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL
    show_progress_bar: bool = True
    model: Any | None = field(default=None, init=False, repr=False)
    catalog: pd.DataFrame = field(default_factory=pd.DataFrame, init=False, repr=False)
    anime_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32), init=False)
    catalog_embeddings: np.ndarray | None = field(default=None, init=False, repr=False)
    item_id_to_index: dict[int, int] = field(default_factory=dict, init=False, repr=False)

    def fit(self, anime_catalog: pd.DataFrame, *_: Any, **__: Any) -> CineSenseV1Recommender:
        """Build catalog tags and precompute catalog embeddings exactly once."""
        self.catalog = build_catalog_tags(anime_catalog)
        self.anime_ids = self.catalog[ITEM_ID_COL].astype(np.int32).to_numpy()
        self.item_id_to_index = {
            int(item_id): index for index, item_id in enumerate(self.anime_ids.tolist())
        }

        self.model = self._load_embedding_model(self.embedding_model_name)
        embeddings = self.model.encode(
            self.catalog["tags"].tolist(),
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self.catalog_embeddings = np.asarray(embeddings, dtype=np.float32)
        return self

    def recommend_for_user(self, user_id: int, train_items: set[int], k: int) -> list[int]:
        """Recommend ranked anime IDs from the average embedding of train items."""
        if k <= 0:
            raise ValueError("k must be greater than 0.")
        if self.catalog_embeddings is None:
            raise RuntimeError("CineSenseV1Recommender must be fitted before recommend().")

        train_indices = [
            self.item_id_to_index[item_id]
            for item_id in train_items
            if item_id in self.item_id_to_index
        ]
        if not train_indices:
            return []

        user_embedding = self.catalog_embeddings[train_indices].mean(axis=0)
        norm = np.linalg.norm(user_embedding)
        if norm == 0.0 or not np.isfinite(norm):
            return []

        user_embedding = user_embedding / norm
        similarity_scores = self.catalog_embeddings @ user_embedding
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

    def _load_embedding_model(self, model_name: str) -> Any:
        try:
            # pyrefly: ignore [missing-import]
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for research/training workflows. "
                "Install requirements-dev.txt."
            ) from exc

        return SentenceTransformer(model_name)


@dataclass
class CineSenseSingleSeedMaxRecommender(CineSenseV1Recommender):
    """Semantic recommender that scores candidates by max similarity to any train item."""

    model_name: str = "cinesense_single_seed_max"
    seed_batch_size: int = 128

    def recommend_for_user(self, user_id: int, train_items: set[int], k: int) -> list[int]:
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


@dataclass
class CineSenseHybridLite(CineSenseSingleSeedMaxRecommender):
    """Single-seed max semantic retrieval with a global train-popularity prior."""

    model_name: str = "cinesense_hybrid_lite"
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
    ) -> CineSenseHybridLite:
        """Fit semantic embeddings and normalized train-only popularity scores."""
        super().fit(anime_catalog)
        self.popularity_scores = self._build_popularity_scores(train_interactions)
        return self

    def recommend_for_user(self, user_id: int, train_items: set[int], k: int) -> list[int]:
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


@dataclass
class CineSenseHybridLiteWeighted(CineSenseHybridLite):
    """Hybrid Lite with rating-weighted single-seed max semantic scoring."""

    model_name: str = "cinesense_hybrid_lite_weighted"
    rating_weight_scheme: RatingWeightScheme = "normalized"
    user_item_weights: dict[int, dict[int, float]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def fit(
        self,
        anime_catalog: pd.DataFrame,
        train_interactions: pd.DataFrame,
        user_ids: Iterable[int] | None = None,
    ) -> CineSenseHybridLiteWeighted:
        """Fit base Hybrid Lite state and user-specific train-item rating weights."""
        super().fit(anime_catalog, train_interactions)
        self.user_item_weights = self._build_user_item_weights(train_interactions, user_ids)
        return self

    def recommend_for_user(self, user_id: int, train_items: set[int], k: int) -> list[int]:
        """Recommend ranked anime IDs using rating-weighted seed similarities."""
        if k <= 0:
            raise ValueError("k must be greater than 0.")
        if self.catalog_embeddings is None:
            raise RuntimeError("CineSenseHybridLiteWeighted must be fitted before recommend().")
        if self.popularity_scores.size != len(self.anime_ids):
            raise RuntimeError("Popularity scores are not initialized correctly.")

        train_indices, train_weights = self._weighted_train_indices(user_id, train_items)
        if train_indices.size == 0:
            return []

        semantic_scores = self._weighted_max_similarity_to_train_items(
            train_indices,
            train_weights,
        )
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

    def _weighted_train_indices(self, user_id: int, train_items: set[int]) -> tuple[np.ndarray, np.ndarray]:
        item_weights = self.user_item_weights.get(user_id, {})
        train_indices = []
        train_weights = []

        for item_id in train_items:
            if item_id not in self.item_id_to_index:
                continue

            train_indices.append(self.item_id_to_index[item_id])
            train_weights.append(item_weights.get(item_id, 1.0))

        return (
            np.asarray(train_indices, dtype=np.int32),
            np.asarray(train_weights, dtype=np.float32),
        )

    def _weighted_max_similarity_to_train_items(
        self,
        train_indices: np.ndarray,
        train_weights: np.ndarray,
    ) -> np.ndarray:
        if self.catalog_embeddings is None:
            raise RuntimeError("CineSenseHybridLiteWeighted must be fitted before scoring.")

        total_weights = np.sum(train_weights)
        if total_weights <= 0:
            return np.zeros(len(self.anime_ids), dtype=np.float32)

        weighted_sim_sum = np.zeros(len(self.anime_ids), dtype=np.float32)
        max_sim = np.full(len(self.anime_ids), -np.inf, dtype=np.float32)
        batch_size = max(1, self.seed_batch_size)

        for start in range(0, train_indices.size, batch_size):
            batch_indices = train_indices[start : start + batch_size]
            batch_weights = train_weights[start : start + batch_size]
            train_embeddings = self.catalog_embeddings[batch_indices]
            
            # Raw similarity
            batch_scores = self.catalog_embeddings @ train_embeddings.T
            
            # Running max of raw similarity
            max_sim = np.maximum(max_sim, batch_scores.max(axis=1))
            
            # Running sum of weighted similarities
            weighted_batch_scores = batch_scores * batch_weights.reshape(1, -1)
            weighted_sim_sum += weighted_batch_scores.sum(axis=1)

        weighted_avg = weighted_sim_sum / total_weights
        hybrid_score = 0.7 * weighted_avg + 0.3 * max_sim
        return hybrid_score

    def _build_user_item_weights(
        self,
        train_interactions: pd.DataFrame,
        user_ids: Iterable[int] | None = None,
    ) -> dict[int, dict[int, float]]:
        user_item_weights: dict[int, dict[int, float]] = {}
        USER_ID_COL = "user_id"
        if user_ids is not None:
            user_id_set = set(user_ids)
            train_interactions = train_interactions[
                train_interactions[USER_ID_COL].isin(user_id_set)
            ]

        for user_id, user_rows in train_interactions.groupby(USER_ID_COL, sort=False):
            user_item_weights[int(user_id)] = {
                int(item_id): self._rating_weight(int(score))
                for item_id, score in zip(
                    user_rows[ITEM_ID_COL].to_numpy(),
                    user_rows["score"].to_numpy(),
                    strict=False,
                )
            }

        return user_item_weights

    def _rating_weight(self, score: int) -> float:
        if self.rating_weight_scheme == "raw_score":
            return float(score)
        if self.rating_weight_scheme == "normalized":
            return float(score) / 10.0
        if self.rating_weight_scheme == "strong":
            if score <= 7:
                return 1.0
            if score == 8:
                return 2.0
            if score == 9:
                return 4.0
            return 8.0

        raise ValueError(f"Unknown rating weight scheme: {self.rating_weight_scheme}")
