from fastapi import Request
from cinesense.services.recommendation import RecommendationService


def get_recommendation_service(request: Request) -> RecommendationService:
    """Dependency provider that fetches the loaded RecommendationService from the app context."""
    return request.app.state.recommendation_service
