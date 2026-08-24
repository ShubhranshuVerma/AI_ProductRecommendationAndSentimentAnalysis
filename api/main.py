from fastapi import FastAPI

from api.routes.recommendation import router as recommendation_router
from api.routes.sentiment import router as sentiment_router
from api.routes.review_analysis import router as review_analysis_router


app = FastAPI(
    title="AI Product Recommendation & Review Analysis System",
    description="AI-powered product recommendation and review analysis API.",
    version="1.0.0",
)


app.include_router(recommendation_router)
app.include_router(sentiment_router)
app.include_router(review_analysis_router)


@app.get("/")
def root():
    return {
        "message": "AI Product Recommendation API is running."
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }