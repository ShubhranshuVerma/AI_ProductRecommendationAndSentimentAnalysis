from pathlib import Path

import mlflow


EXPERIMENT_NAME = "AI_Product_Recommendation_Sentiment"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"


def configure_mlflow():
    """Configure the local MLflow tracking store."""

    mlflow.set_tracking_uri(
        f"sqlite:///{MLFLOW_DB}"
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )


def start_run(model_name):
    return mlflow.start_run(
        run_name=model_name
    )