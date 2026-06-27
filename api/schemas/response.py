from pydantic import BaseModel, Field


class MatchedSeedResponse(BaseModel):
    anime_id: int
    title: str


class SeedShareResponse(BaseModel):
    title: str
    share: float


class ExplanationResponse(BaseModel):
    matched_seed: MatchedSeedResponse | None = None
    similarity: float | None = Field(None, description="Cosine similarity score.")
    popularity: float | None = Field(None, description="Popularity score.")
    summary: str = Field(..., description="Human-readable summary explanation.")
    reasons: list[str] = Field(default_factory=list, description="List of reasons for recommendation.")
    reason: str | None = Field(None, description="Legacy single string reason for backward compatibility.")
    seed_shares: dict[int, SeedShareResponse] | None = Field(None, description="Contribution shares of each seed anime.")


class RecommendedAnime(BaseModel):
    anime_id: int
    title: str
    title_english: str | None = None
    score: float = Field(..., description="Calculated final score of the recommendation.")
    match_score: float = Field(..., description="User-friendly match score out of 10.")
    match_badge: str = Field(..., description="Qualitative match badge.")
    explanation: ExplanationResponse = Field(
        ..., description="Structured explanation of why this item is recommended."
    )


class RecommendResponse(BaseModel):
    recommendations: list[RecommendedAnime]
