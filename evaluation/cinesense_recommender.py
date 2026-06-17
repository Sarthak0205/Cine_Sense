from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from evaluation.datasets import ENGLISH_TITLE_COL, ITEM_ID_COL, SYNOPSIS_COL, TITLE_COL


DEFAULT_MODEL_NAME = "cinesense_v1"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@dataclass
class CineSenseV1Recommender:
    """Semantic CineSense recommender adapted to the benchmark interface."""

    model_name: str = DEFAULT_MODEL_NAME
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL
    show_progress_bar: bool = True
    model: Any | None = field(default=None, init=False, repr=False)
    catalog: pd.DataFrame = field(default_factory=pd.DataFrame, init=False, repr=False)
    anime_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32), init=False)
    catalog_embeddings: np.ndarray | None = field(default=None, init=False, repr=False)
    item_id_to_index: dict[int, int] = field(default_factory=dict, init=False, repr=False)

    def fit(self, anime_catalog: pd.DataFrame, *_: Any, **__: Any) -> "CineSenseV1Recommender":
        """Build catalog tags and precompute catalog embeddings exactly once."""

        self.catalog = _build_catalog_tags(anime_catalog)
        self.anime_ids = self.catalog[ITEM_ID_COL].astype(np.int32).to_numpy()
        self.item_id_to_index = {
            int(item_id): index for index, item_id in enumerate(self.anime_ids.tolist())
        }

        self.model = _load_embedding_model(self.embedding_model_name)
        embeddings = self.model.encode(
            self.catalog["tags"].tolist(),
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self.catalog_embeddings = np.asarray(embeddings, dtype=np.float32)
        return self

    def recommend(self, user_id: int, train_items: set[int], k: int) -> list[int]:
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


def _build_catalog_tags(anime_catalog: pd.DataFrame) -> pd.DataFrame:
    catalog = anime_catalog[[ITEM_ID_COL, TITLE_COL, ENGLISH_TITLE_COL, SYNOPSIS_COL]].copy()
    catalog.dropna(subset=[ITEM_ID_COL, TITLE_COL, SYNOPSIS_COL], inplace=True)
    catalog.drop_duplicates(subset=[ITEM_ID_COL], keep="first", inplace=True, ignore_index=True)

    catalog[TITLE_COL] = catalog[TITLE_COL].astype(str).str.lower()
    catalog[ENGLISH_TITLE_COL] = catalog[ENGLISH_TITLE_COL].fillna("").astype(str).str.lower()
    catalog[SYNOPSIS_COL] = catalog[SYNOPSIS_COL].astype(str).map(_clean_text)
    catalog["tags"] = (
        catalog[TITLE_COL]
        + " "
        + catalog[ENGLISH_TITLE_COL]
        + " "
        + catalog[SYNOPSIS_COL]
    )

    return catalog[[ITEM_ID_COL, TITLE_COL, ENGLISH_TITLE_COL, SYNOPSIS_COL, "tags"]]


def _clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _load_embedding_model(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)
