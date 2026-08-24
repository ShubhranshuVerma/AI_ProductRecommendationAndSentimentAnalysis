from fastapi import APIRouter, HTTPException

from api.schemas import (
    RecommendationRequest,
    RecommendationResponse,
)
from recommendation.service import get_recommendations


router = APIRouter(
    prefix="/recommend",
    tags=["Recommendation"],
)


@router.post(
    "",
    response_model=RecommendationResponse,
)
def recommend_products(
    request: RecommendationRequest,
):
    try:
        recommendations = get_recommendations(
            request.customer_id,
            request.top_n,
        )

        items = recommendations.to_dict(
            orient="records"
        )

        return {
            "customer_id": request.customer_id,
            "recommendations": items,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )