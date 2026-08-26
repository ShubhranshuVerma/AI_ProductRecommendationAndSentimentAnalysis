from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    customer_id: str = Field(..., min_length=1)
    top_n: int = Field(default=10, ge=1, le=50)


class RecommendationItem(BaseModel):
    product_id: str
    product_name: str
    category: str
    recommendation_score: float


class RecommendationResponse(BaseModel):
    customer_id: str
    recommendations: list[RecommendationItem]

class SentimentRequest(BaseModel):
    review_text: str = Field(..., min_length=1, max_length=5000)


class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float | None = None
    model: str | None = None

class ReviewItem(BaseModel):
    sentiment: str = Field(..., min_length=1)
    review_text: str = Field(..., min_length=1, max_length=5000)


class ReviewAnalysisRequest(BaseModel):
    product_name: str = Field(..., min_length=1, max_length=500)
    reviews: list[ReviewItem] = Field(..., min_length=1, max_length=50)


class ReviewAnalysisResponse(BaseModel):
    summary: str
    common_complaints: list[str]
    praised_features: list[str]
    business_insights: list[str]