from fastapi import APIRouter, Depends, HTTPException, status
from api.schemas.request import RecommendRequest
from api.schemas.response import RecommendResponse
from api.dependencies import get_recommendation_service
from cinesense.services.recommendation import RecommendationService

router = APIRouter()


@router.post(
    "/recommend",
    response_model=RecommendResponse,
    status_code=status.HTTP_200_OK,
    summary="Get personalized recommendations",
    description="Stateless endpoint that generates recommendations given a set of seed IDs and optional ratings.",
)
def get_recommendations(
    request: RecommendRequest,
    service: RecommendationService = Depends(get_recommendation_service),
):
    try:
        # Validate inputs first (raising ValueError/TypeError if input format or bounds are wrong)
        valid_ids, validated_ratings = service.validate_inputs(
            request.anime_ids,
            request.ratings,
            request.top_k,
        )

        # If no valid IDs remain after checking, return empty recommendation list
        if not valid_ids:
            return RecommendResponse(recommendations=[])

        # Get recommendations
        raw_recs = service.recommend(
            anime_ids=valid_ids,
            ratings=validated_ratings,
            top_k=request.top_k,
            mode=request.mode,
            user_id=request.user_id,
        )

        # Map to response schema
        formatted_recs = []
        for item in raw_recs:
            formatted_recs.append({
                "anime_id": item["anime_id"],
                "title": item["title"],
                "title_english": item["title_english"],
                "score": round(item["score"], 4),
                "explanation": item["explanation"],
            })

        return RecommendResponse(recommendations=formatted_recs)

    except (TypeError, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal error occurred: {str(e)}",
        )
