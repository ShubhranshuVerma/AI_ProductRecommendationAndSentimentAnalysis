import sys
import time
from pathlib import Path

import mlflow
import mlflow.pyfunc
from sentiment.transformer_service import (
    MODEL_NAME,
    get_sentiment_pipeline,
)
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)
class RoBERTaModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        from transformers import pipeline

        self.model = pipeline(
            "sentiment-analysis",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            device=-1,
        )

    def predict(self, context, model_input):

        if hasattr(model_input, "tolist"):
            texts = model_input.tolist()
        else:
            texts = list(model_input)

        results = self.model(
            texts,
            truncation=True,
            max_length=512,
        )

        return [
            result["label"].lower()
            for result in results
        ]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATA_DIR = Path(
    "data/processed/sentiment"
)

EXPERIMENT_NAME = (
    "AI_Product_Recommendation_Sentiment"
)

REGISTERED_MODEL_NAME = (
    "ProductSentimentModel"
)

MODEL_NAME = (
    "cardiffnlp/"
    "twitter-roberta-base-sentiment-latest"
)


def load_data():
    train_df = pd.read_csv(
        DATA_DIR / "train.csv"
    )

    test_df = pd.read_csv(
        DATA_DIR / "test.csv"
    )

    return train_df, test_df


def evaluate_roberta(test_df):
    from sentiment.transformer_service import (
        get_sentiment_pipeline,
    )

    model = get_sentiment_pipeline()

    texts = test_df["review_text"].tolist()
    actual = test_df["sentiment"].tolist()

    start_time = time.perf_counter()

    predictions = [
        result["label"].lower()
        for result in model(
            texts,
            truncation=True,
            max_length=512,
        )
    ]

    inference_time = (
        time.perf_counter()
        - start_time
    )

    metrics = {
        "accuracy": accuracy_score(
            actual,
            predictions,
        ),
        "macro_f1": f1_score(
            actual,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            actual,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "negative_recall": recall_score(
            actual,
            predictions,
            labels=["negative"],
            average="macro",
            zero_division=0,
        ),
        "inference_time_seconds": (
            inference_time
        ),
    }

    return metrics, predictions


def main():

    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    print(
        "Loading sentiment dataset..."
    )

    _, test_df = load_data()

    print(
        f"Testing samples: "
        f"{len(test_df):,}"
    )

    print()
    print("=" * 70)
    print("FINAL MODEL")
    print("=" * 70)

    print(
        f"Model: {MODEL_NAME}"
    )

    metrics, predictions = (
        evaluate_roberta(test_df)
    )

    print()
    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Macro F1: "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{metrics['weighted_f1']:.4f}"
    )

    print(
        f"Negative Recall: "
        f"{metrics['negative_recall']:.4f}"
    )

    print(
        f"Inference Time: "
        f"{metrics['inference_time_seconds']:.4f}s"
    )

    print()
    print(
        classification_report(
            test_df["sentiment"],
            predictions,
            zero_division=0,
        )
    )

    print()
    print("=" * 70)
    print("MLFLOW REGISTRATION")
    print("=" * 70)

    with mlflow.start_run(
        run_name="final_twitter_roberta"
    ):

        mlflow.log_param(
            "model",
            "twitter_roberta",
        )

        mlflow.log_param(
            "model_name",
            MODEL_NAME,
        )

        mlflow.log_param(
            "model_type",
            "pretrained_transformer",
        )

        mlflow.log_param(
            "dataset",
            "real_english_sentiment_reviews",
        )

        for name, value in metrics.items():
            mlflow.log_metric(
                name,
                float(value),
            )

        mlflow.set_tag(
            "task",
            "sentiment_classification",
        )

        mlflow.set_tag(
            "model_selection",
            "best_generalization",
        )

        mlflow.set_tag(
            "production_model",
            "true",
        )

        mlflow.set_tag(
            "framework",
            "huggingface_transformers",
        )

        # Register the Hugging Face model reference.
        model_info = mlflow.pyfunc.log_model(
            python_model=RoBERTaModel(),
            name="sentiment_model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print(
            f"Model URI: "
            f"{model_info.model_uri}"
        )

        print(
            f"Registered model: "
            f"{REGISTERED_MODEL_NAME}"
        )

        print(
            f"Model URI: "
            f"{model_info.model_uri}"
        )

        print(
            f"Registered model: "
            f"{REGISTERED_MODEL_NAME}"
        )

    print()
    print("=" * 70)
    print(
        "MODEL REGISTRATION COMPLETED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()