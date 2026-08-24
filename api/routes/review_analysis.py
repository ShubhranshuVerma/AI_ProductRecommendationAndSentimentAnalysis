from fastapi import APIRouter, HTTPException
from api.schemas import ReviewAnalysisRequest, ReviewAnalysisResponse
from llm.review_analysis import analyze_reviews

router = APIRouter(prefix="/review-analysis", tags=["Review Analysis"])

@router.post("", response_model=ReviewAnalysisResponse)
def review_analysis(request: ReviewAnalysisRequest):
    try:
        reviews = [review.model_dump() for review in request.reviews]
        return analyze_reviews(request.product_name, reviews)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Review analysis failed: {error}")