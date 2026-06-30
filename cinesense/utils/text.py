import re
import pandas as pd

ITEM_ID_COL = "anime_id"
TITLE_COL = "title"
ENGLISH_TITLE_COL = "title_english"
SYNOPSIS_COL = "synopsis"


def clean_text(text: str) -> str:
    """Lowercases text, replaces non-alphabetic characters with spaces, and cleans whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_catalog_tags(anime_catalog: pd.DataFrame) -> pd.DataFrame:
    """Cleans catalog columns, drop duplicates, and generates textual tag summaries."""
    catalog = anime_catalog[[ITEM_ID_COL, TITLE_COL, ENGLISH_TITLE_COL, SYNOPSIS_COL]].copy()
    catalog.dropna(subset=[ITEM_ID_COL, TITLE_COL, SYNOPSIS_COL], inplace=True)
    catalog.drop_duplicates(subset=[ITEM_ID_COL], keep="first", inplace=True, ignore_index=True)

    catalog[TITLE_COL] = catalog[TITLE_COL].astype(str).str.lower()
    catalog[ENGLISH_TITLE_COL] = catalog[ENGLISH_TITLE_COL].fillna("").astype(str).str.lower()
    catalog[SYNOPSIS_COL] = catalog[SYNOPSIS_COL].astype(str).map(clean_text)
    catalog["tags"] = (
        catalog[TITLE_COL]
        + " "
        + catalog[ENGLISH_TITLE_COL]
        + " "
        + catalog[SYNOPSIS_COL]
    )

    return catalog[[ITEM_ID_COL, TITLE_COL, ENGLISH_TITLE_COL, SYNOPSIS_COL, "tags"]]


def normalize_synopsis(synopsis: str | None) -> str:
    """Safely normalizes synopsis text by trimming and collapsing multiple spaces/newlines."""
    if not synopsis or not isinstance(synopsis, str):
        return "No synopsis available for this title."
    cleaned = " ".join(synopsis.split())
    if cleaned.lower() in ("", "none", "null"):
        return "No synopsis available for this title."
    return cleaned

