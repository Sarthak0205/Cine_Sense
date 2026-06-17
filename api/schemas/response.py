from pydantic import BaseModel, Field


class MatchedSeedResponse(BaseModel):
    anime_id: int
    title: str


class ExplanationResponse(BaseModel):
    matched_seed: MatchedSeedResponse | None = None
    similarity: float | None = Field(None, description="Cosine similarity score.")
    popularity: float | None = Field(None, description="Popularity score.")
    summary: str = Field(..., description="Human-readable summary explanation.")
    reasons: list[str] = Field(default_factory=list, description="List of reasons for recommendation.")
    reason: str | None = Field(None, description="Legacy single string reason for backward compatibility.")


class RecommendedAnime(BaseModel):
    anime_id: int
    title: str
    title_english: str | None = None
    score: float = Field(..., description="Calculated final score of the recommendation.")
    explanation: ExplanationResponse = Field(
        ..., description="Structured explanation of why this item is recommended."
    )


class RecommendResponse(BaseModel):
    recommendations: list[RecommendedAnime]
