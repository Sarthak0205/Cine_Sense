import logging
from time import perf_counter
from fastapi import APIRouter, Depends, HTTPException, status
from api.schemas.request import RecommendRequest
from api.schemas.response import RecommendResponse
from api.dependencies import get_recommendation_service
from cinesense.services.recommendation import RecommendationService

logger = logging.getLogger(__name__)

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
    t_start_request = perf_counter()
    logger.info(
        "request_received",
        extra={
            "anime_ids": request.anime_ids,
            "top_k": request.top_k,
            "mode": request.mode,
            "user_id": request.user_id,
        }
    )

    try:
        # Validate inputs first (raising ValueError/TypeError if input format or bounds are wrong)
        valid_ids, validated_ratings = service.validate_inputs(
            request.anime_ids,
            request.ratings,
            request.top_k,
        )

        # If no valid IDs remain after checking, return empty recommendation list
        if not valid_ids:
            duration_ms = (perf_counter() - t_start_request) * 1000.0
            logger.info(
                "recommendation_completed",
                extra={
                    "anime_count": 0,
                    "top_k": request.top_k,
                    "duration_ms": duration_ms
                }
            )
            return RecommendResponse(recommendations=[])

        # Get recommendations
        t_start_inference = perf_counter()
        logger.info(
            "recommendation_generation_start",
            extra={"anime_ids": valid_ids}
        )

        raw_recs = service.recommend(
            anime_ids=valid_ids,
            ratings=validated_ratings,
            top_k=request.top_k,
            mode=request.mode,
            user_id=request.user_id,
        )

        t_end_inference = perf_counter()
        inference_ms = (t_end_inference - t_start_inference) * 1000.0
        logger.info(
            "recommendation_generation_end",
            extra={"duration_ms": inference_ms}
        )

        # Map to response schema
        formatted_recs = []
        for item in raw_recs:
            formatted_recs.append({
                "anime_id": item["anime_id"],
                "title": item["title"],
                "title_english": item["title_english"],
                "score": round(item["score"], 4),
                "match_score": item["match_score"],
                "match_badge": item["match_badge"],
                "explanation": item["explanation"],
            })

        duration_ms = (perf_counter() - t_start_request) * 1000.0
        logger.info(
            "recommendation_completed",
            extra={
                "anime_count": len(valid_ids),
                "top_k": request.top_k,
                "duration_ms": duration_ms
            }
        )

        return RecommendResponse(recommendations=formatted_recs)

    except (TypeError, ValueError) as e:
        duration_ms = (perf_counter() - t_start_request) * 1000.0
        logger.error(
            "recommendation_failed_validation",
            extra={
                "error": str(e),
                "duration_ms": duration_ms
            }
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        duration_ms = (perf_counter() - t_start_request) * 1000.0
        logger.error(
            "recommendation_failed_internal",
            extra={
                "error": str(e),
                "duration_ms": duration_ms
            }
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal error occurred: {str(e)}",
        )
