import os

from dotenv import load_dotenv


load_dotenv()


APP_NAME = os.getenv(
    "APP_NAME",
    "AI Product Recommendation & Review Analysis System"
)

APP_ENV = os.getenv(
    "APP_ENV",
    "development"
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///mlflow.db"
)

MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "product_review_sentiment"
)

LLM_API_KEY = os.getenv("LLM_API_KEY")