from pathlib import Path

import joblib
from fastapi import APIRouter, HTTPException

from api.schemas import SentimentRequest, SentimentResponse


router = APIRouter(
    prefix="/sentiment",
    tags=["Sentiment"],
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "sentiment_model.joblib"
)

_model = None


def get_model():
    global _model

    if _model is None:
        try:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Sentiment model not found: {MODEL_PATH}"
                )

            _model = joblib.load(MODEL_PATH)

        except Exception as error:
            raise RuntimeError(
                f"Unable to load sentiment model: {error}"
            )

    return _model


@router.post(
    "",
    response_model=SentimentResponse,
)
def predict_sentiment(
    request: SentimentRequest,
):
    try:
        model = get_model()

        prediction = model.predict(
            [request.review_text]
        )

        sentiment = str(prediction[0])

        return {
            "sentiment": sentiment
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Sentiment prediction failed: {error}"
            ),
        )