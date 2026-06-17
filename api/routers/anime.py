from fastapi import APIRouter, Depends, HTTPException, Query, status
from api.dependencies import get_recommendation_service
from cinesense.services.recommendation import RecommendationService

router = APIRouter(prefix="/anime", tags=["discovery"])


@router.get(
    "/search",
    status_code=status.HTTP_200_OK,
    summary="Search for anime by title",
    description="Case-insensitive, partial substring search matching both english and original titles.",
)
def search_anime(
    q: str = Query(..., min_length=1, description="Search query string"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of search results to return"),
    service: RecommendationService = Depends(get_recommendation_service),
):
    try:
        results = service.search_anime(query=q, limit=limit)
        return {"results": results}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal error occurred: {str(e)}",
        )


@router.get(
    "/{anime_id}",
    status_code=status.HTTP_200_OK,
    summary="Get anime details by ID",
    description="Look up full details (title, title_english, synopsis) for a specific anime ID.",
)
def get_anime_details(
    anime_id: int,
    service: RecommendationService = Depends(get_recommendation_service),
):
    try:
        details = service.get_anime_details(anime_id=anime_id)
        if details is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Anime with ID {anime_id} not found in the catalog.",
            )
        return details
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal error occurred: {str(e)}",
        )
