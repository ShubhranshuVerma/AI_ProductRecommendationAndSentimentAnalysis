from fastapi import APIRouter, HTTPException

from api.schemas import (
    SentimentRequest,
    SentimentResponse,
)

from sentiment.transformer_service import (
    predict_sentiment as predict_roberta_sentiment,
)


router = APIRouter(
    prefix="/sentiment",
    tags=["Sentiment"],
)


@router.post(
    "",
    response_model=SentimentResponse,
)
def predict_sentiment(
    request: SentimentRequest,
):
    try:
        result = predict_roberta_sentiment(
            request.review_text
        )

        return {
            "sentiment": result["sentiment"],
            "confidence": result["confidence"],
            "model": (
                "cardiffnlp/"
                "twitter-roberta-base-sentiment-latest"
            ),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Sentiment prediction failed: {error}"
            ),
        )