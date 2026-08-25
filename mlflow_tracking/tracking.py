from pathlib import Path

import mlflow


# ============================================================
# MLflow configuration
# ============================================================

EXPERIMENT_NAME = (
    "AI_Product_Recommendation_Sentiment"
)

ARTIFACT_DIR = Path("mlruns")


def configure_mlflow():
    """
    Configure the local MLflow tracking store
    and select the project experiment.
    """

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )


def start_run(model_name):
    """
    Start an MLflow run for a candidate model.
    """

    return mlflow.start_run(
        run_name=model_name
    )