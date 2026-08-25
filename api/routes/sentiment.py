import os
from pathlib import Path

import mlflow
from fastapi import APIRouter, HTTPException

from api.schemas import SentimentRequest, SentimentResponse


router = APIRouter(
    prefix="/sentiment",
    tags=["Sentiment"],
)


MODEL_ID = "m-6d846e15ec80497ebac6df74e880238f"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOCAL_MODEL_PATH = (
    PROJECT_ROOT
    / "mlruns"
    / "1"
    / "models"
    / MODEL_ID
    / "artifacts"
)

DOCKER_MODEL_PATH = (
    Path("/app")
    / "mlruns"
    / "1"
    / "models"
    / MODEL_ID
    / "artifacts"
)


if DOCKER_MODEL_PATH.exists():
    DEFAULT_MODEL_URI = str(DOCKER_MODEL_PATH)
else:
    DEFAULT_MODEL_URI = str(LOCAL_MODEL_PATH)


MODEL_URI = os.getenv(
    "SENTIMENT_MODEL_URI",
    DEFAULT_MODEL_URI,
)


_model = None


def get_model():
    global _model

    if _model is None:
        try:
            _model = mlflow.pyfunc.load_model(MODEL_URI)
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

        return {
            "sentiment": str(prediction[0])
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Sentiment prediction failed: {error}",
        )