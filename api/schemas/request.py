from typing import Literal
from pydantic import BaseModel, Field, field_validator


class RecommendRequest(BaseModel):
    anime_ids: list[int] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of watched anime IDs (seeds).",
        examples=[1535, 5114, 9253],
    )
    ratings: dict[int, float] | None = Field(
        None,
        description="Optional ratings dictionary mapping anime_id -> rating (1.0 to 10.0).",
        examples=[{"1535": 10.0, "5114": 9.0, "9253": 8.0}],
    )
    top_k: int = Field(
        10,
        ge=1,
        le=100,
        description="Number of recommendations to return.",
    )
    mode: Literal["discover", "similar"] = Field(
        "discover",
        description="Recommendation path mode: 'discover' (default) or 'similar' (sequels mode).",
    )
    user_id: str | None = Field(
        default=None,
        max_length=100,
        description="Optional user ID for deterministic A/B traffic routing.",
        examples=["user_12345"],
    )

    @field_validator("user_id")
    @classmethod
    def validate_and_normalize_user_id(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v_stripped = v.strip()
        if not v_stripped:
            raise ValueError("user_id cannot be empty or whitespace-only.")
        return v_stripped
