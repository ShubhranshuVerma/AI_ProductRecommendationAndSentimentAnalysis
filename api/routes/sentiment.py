import mlflow
from fastapi import APIRouter, HTTPException

from api.schemas import SentimentRequest, SentimentResponse


router = APIRouter(prefix="/sentiment", tags=["Sentiment"])

import os

import os

MODEL_URI = os.getenv(
    "SENTIMENT_MODEL_URI",
    "models:/ProductSentimentModel/2",
)

_model = None


def get_model():
    global _model

    if _model is None:
        try:
            _model = mlflow.pyfunc.load_model(MODEL_URI)
        except Exception as error:
            raise RuntimeError(
                f"Unable to load registered sentiment model: {error}"
            )

    return _model


@router.post("", response_model=SentimentResponse)
def predict_sentiment(request: SentimentRequest):
    try:
        model = get_model()
        prediction = model.predict([request.review_text])
        sentiment = str(prediction[0])

        return {"sentiment": sentiment}

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Sentiment prediction failed: {error}",
        )