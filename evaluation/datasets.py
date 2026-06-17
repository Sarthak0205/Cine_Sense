from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_USER_WATCHES_PATH = Path("archive-2/user_watches.csv")
DEFAULT_ANIME_CATALOG_PATH = Path("archive-2/animes.csv")
DEFAULT_SPLITS_DIR = Path("evaluation/splits")

DEFAULT_MIN_SCORE = 7
DEFAULT_MIN_POSITIVE_INTERACTIONS = 20

DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1
DEFAULT_RANDOM_SEED = 42

USER_ID_COL = "user_id"
ITEM_ID_COL = "anime_id"
SCORE_COL = "score"
STATUS_COL = "status"
WATCHED_EPISODES_COL = "num_watched_episodes"

TITLE_COL = "title"
ENGLISH_TITLE_COL = "title_english"
SYNOPSIS_COL = "synopsis"
AIRING_STATUS_COL = "airing_status"
MPAA_RATING_COL = "mpaa_rating"
NUM_EPISODES_COL = "num_episodes"
IMAGE_PATH_COL = "image_path"

USER_WATCHES_COLUMNS = [
    ITEM_ID_COL,
    SCORE_COL,
    STATUS_COL,
    WATCHED_EPISODES_COL,
    USER_ID_COL,
]

ANIME_CATALOG_COLUMNS = [
    TITLE_COL,
    ITEM_ID_COL,
    IMAGE_PATH_COL,
    AIRING_STATUS_COL,
    NUM_EPISODES_COL,
    MPAA_RATING_COL,
    SYNOPSIS_COL,
    ENGLISH_TITLE_COL,
]


@dataclass(frozen=True)
class DatasetSplit:
    """Per-user train, validation, and test interaction split."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    eligible_user_ids: list[int]
    catalog_item_ids: set[int]
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float


@dataclass(frozen=True)
class EvalUser:
    """Benchmark-ready view of one user's split interactions."""

    user_id: int
    train_items: set[int]
    validation_items: set[int]
    test_items: set[int]


def load_user_watches(path: str | Path = DEFAULT_USER_WATCHES_PATH) -> pd.DataFrame:
    """Load raw user-anime interactions using only evaluation-relevant columns."""

    user_watches = pd.read_csv(
        path,
        usecols=USER_WATCHES_COLUMNS,
        dtype={
            USER_ID_COL: "int32",
            ITEM_ID_COL: "int32",
            SCORE_COL: "int8",
            STATUS_COL: "int8",
            WATCHED_EPISODES_COL: "int16",
        },
    )
    user_watches.dropna(subset=[USER_ID_COL, ITEM_ID_COL], inplace=True)
    user_watches.drop_duplicates(inplace=True, ignore_index=True)
    return user_watches


def load_anime_catalog(path: str | Path = DEFAULT_ANIME_CATALOG_PATH) -> pd.DataFrame:
    """Load anime catalog metadata and return one row per anime ID."""

    catalog = pd.read_csv(
        path,
        usecols=ANIME_CATALOG_COLUMNS,
        dtype={
            ITEM_ID_COL: "int32",
            TITLE_COL: "string",
            ENGLISH_TITLE_COL: "string",
            SYNOPSIS_COL: "string",
            IMAGE_PATH_COL: "string",
            AIRING_STATUS_COL: "Int8",
            NUM_EPISODES_COL: "Int16",
            MPAA_RATING_COL: "string",
        },
    )
    catalog.dropna(subset=[ITEM_ID_COL, TITLE_COL], inplace=True)
    catalog.drop_duplicates(subset=[ITEM_ID_COL], keep="first", inplace=True, ignore_index=True)
    return catalog


def build_positive_interactions(
    user_watches: pd.DataFrame,
    min_score: int = DEFAULT_MIN_SCORE,
    catalog_item_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Build positive user-item interactions from ratings and optional catalog membership."""

    positive_mask = user_watches[SCORE_COL] >= min_score

    if catalog_item_ids is not None:
        catalog_ids = set(catalog_item_ids)
        positive_mask &= user_watches[ITEM_ID_COL].isin(catalog_ids)

    positives = user_watches.loc[positive_mask, [USER_ID_COL, ITEM_ID_COL, SCORE_COL]]
    positives = positives.sort_values(SCORE_COL, ascending=False, kind="mergesort")
    positives = positives.drop_duplicates(
        subset=[USER_ID_COL, ITEM_ID_COL],
        keep="first",
        ignore_index=True,
    )
    return positives.sort_values([USER_ID_COL, ITEM_ID_COL], ignore_index=True)


def filter_users(
    positive_interactions: pd.DataFrame,
    min_positive_interactions: int = DEFAULT_MIN_POSITIVE_INTERACTIONS,
) -> pd.DataFrame:
    """Keep users with enough positive interactions for reliable per-user splitting."""

    user_counts = positive_interactions[USER_ID_COL].value_counts(sort=False)
    eligible_user_ids = user_counts.index[user_counts >= min_positive_interactions]
    return positive_interactions.loc[
        positive_interactions[USER_ID_COL].isin(eligible_user_ids)
    ].reset_index(drop=True)


def split_user_interactions(
    interactions: pd.DataFrame,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_RANDOM_SEED,
) -> DatasetSplit:
    """Create reproducible per-user train, validation, and test splits."""

    _validate_split_ratios(train_ratio, val_ratio, test_ratio)

    ordered = interactions.sort_values([USER_ID_COL, ITEM_ID_COL], kind="mergesort")
    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for user_id, user_rows in ordered.groupby(USER_ID_COL, sort=True):
        shuffled = user_rows.sample(frac=1.0, random_state=_stable_user_seed(seed, int(user_id)))
        item_count = len(shuffled)
        n_test = max(1, round(item_count * test_ratio))
        n_val = max(1, round(item_count * val_ratio))

        if item_count - n_test - n_val < 1:
            continue

        test_parts.append(shuffled.iloc[:n_test])
        validation_parts.append(shuffled.iloc[n_test : n_test + n_val])
        train_parts.append(shuffled.iloc[n_test + n_val :])

    train = _concat_split_parts(train_parts)
    validation = _concat_split_parts(validation_parts)
    test = _concat_split_parts(test_parts)
    eligible_user_ids = sorted(train[USER_ID_COL].unique().astype(int).tolist())
    catalog_item_ids = set(interactions[ITEM_ID_COL].unique().astype(int).tolist())

    return DatasetSplit(
        train=train,
        validation=validation,
        test=test,
        eligible_user_ids=eligible_user_ids,
        catalog_item_ids=catalog_item_ids,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )


def build_eval_users(split: DatasetSplit, use_validation: bool = False) -> list[EvalUser]:
    """Convert split DataFrames into benchmark-ready user records."""

    train_by_user = _items_by_user(split.train)
    validation_by_user = _items_by_user(split.validation)
    test_by_user = _items_by_user(split.test)
    target_by_user = validation_by_user if use_validation else test_by_user

    eval_users: list[EvalUser] = []
    for user_id in split.eligible_user_ids:
        if user_id not in train_by_user or user_id not in target_by_user:
            continue

        eval_users.append(
            EvalUser(
                user_id=user_id,
                train_items=train_by_user[user_id],
                validation_items=validation_by_user.get(user_id, set()),
                test_items=test_by_user.get(user_id, set()),
            )
        )

    return eval_users


def _validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("Split ratios must all be positive.")
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum:.6f}.")


def _stable_user_seed(seed: int, user_id: int) -> int:
    return (seed * 1_000_003 + user_id) % (2**32 - 1)


def _concat_split_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=[USER_ID_COL, ITEM_ID_COL, SCORE_COL]).astype(
            {
                USER_ID_COL: "int32",
                ITEM_ID_COL: "int32",
                SCORE_COL: "int8",
            }
        )

    return pd.concat(parts, ignore_index=True).sort_values(
        [USER_ID_COL, ITEM_ID_COL],
        ignore_index=True,
        kind="mergesort",
    )


def _items_by_user(interactions: pd.DataFrame) -> dict[int, set[int]]:
    return {
        int(user_id): set(user_rows[ITEM_ID_COL].astype(int).tolist())
        for user_id, user_rows in interactions.groupby(USER_ID_COL, sort=False)
    }


if __name__ == "__main__":
    anime_catalog = load_anime_catalog()
    user_watch_data = load_user_watches()
    positive_data = build_positive_interactions(
        user_watch_data,
        catalog_item_ids=anime_catalog[ITEM_ID_COL].unique(),
    )
    filtered_positive_data = filter_users(positive_data)
    dataset_split = split_user_interactions(filtered_positive_data)

    print(f"total interactions: {len(user_watch_data)}")
    print(f"positive interactions: {len(positive_data)}")
    print(f"eligible users: {len(dataset_split.eligible_user_ids)}")
    print(f"train size: {len(dataset_split.train)}")
    print(f"validation size: {len(dataset_split.validation)}")
    print(f"test size: {len(dataset_split.test)}")
    print(f"catalog size: {len(anime_catalog)}")
