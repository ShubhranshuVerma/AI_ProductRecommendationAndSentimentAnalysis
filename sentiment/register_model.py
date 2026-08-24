import sys
from pathlib import Path

# ------------------------------------------------------------
# Add project root to Python path
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path(
    "data/processed/sentiment"
)

EXPERIMENT_NAME = (
    "AI_Product_Recommendation_Sentiment"
)

REGISTERED_MODEL_NAME = (
    "ProductSentimentModel"
)

RANDOM_STATE = 42


# ============================================================
# Load data
# ============================================================

def load_data():

    train_df = pd.read_csv(
        DATA_DIR / "train.csv"
    )

    test_df = pd.read_csv(
        DATA_DIR / "test.csv"
    )

    X_train = train_df[
        "review_text"
    ]

    y_train = train_df[
        "sentiment"
    ]

    X_test = test_df[
        "review_text"
    ]

    y_test = test_df[
        "sentiment"
    ]

    return (
        X_train,
        y_train,
        X_test,
        y_test,
    )


# ============================================================
# Create final model pipeline
# ============================================================

def create_final_pipeline():

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),

            (
                "classifier",
                LinearSVC(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return pipeline


# ============================================================
# Evaluate model
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
):

    predictions = model.predict(
        X_test
    )

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),

        "weighted_precision": precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),

        "weighted_recall": recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),

        "weighted_f1": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),

        "macro_f1": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),

        "negative_f1": f1_score(
            y_test,
            predictions,
            labels=["negative"],
            average="macro",
            zero_division=0,
        ),

        "negative_recall": recall_score(
            y_test,
            predictions,
            labels=["negative"],
            average="macro",
            zero_division=0,
        ),
    }

    print()
    print("=" * 70)
    print("FINAL MODEL EVALUATION")
    print("=" * 70)

    for name, value in metrics.items():

        print(
            f"{name:<25}: {value:.4f}"
        )

    print()
    print("Classification Report")
    print("-" * 70)

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    return metrics


# ============================================================
# Configure MLflow
# ============================================================

def configure_mlflow():

    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )


# ============================================================
# Train, log and register model
# ============================================================

def train_and_register():

    print(
        "Loading sentiment dataset..."
    )

    (
        X_train,
        y_train,
        X_test,
        y_test,
    ) = load_data()

    print(
        f"Training samples: "
        f"{len(X_train):,}"
    )

    print(
        f"Testing samples: "
        f"{len(X_test):,}"
    )

    print()
    print(
        "Creating final sentiment pipeline..."
    )

    model = create_final_pipeline()

    print()
    print(
        "Training Balanced Linear SVM..."
    )

    model.fit(
        X_train,
        y_train,
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "MLFLOW REGISTRATION"
    )

    print(
        "=" * 70
    )

    with mlflow.start_run(
        run_name="final_balanced_linear_svm_pipeline"
    ):

        # ----------------------------------------------------
        # Log model parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "model_type",
            "LinearSVC",
        )

        mlflow.log_param(
            "class_weight",
            "balanced",
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE,
        )

        mlflow.log_param(
            "tfidf_ngram_range",
            "(1, 2)",
        )

        mlflow.log_param(
            "tfidf_min_df",
            2,
        )

        mlflow.log_param(
            "tfidf_max_df",
            0.95,
        )

        mlflow.log_param(
            "tfidf_sublinear_tf",
            True,
        )

        # ----------------------------------------------------
        # Log metrics
        # ----------------------------------------------------

        for name, value in metrics.items():

            mlflow.log_metric(
                name,
                float(value),
            )

        # ----------------------------------------------------
        # Tags
        # ----------------------------------------------------

        mlflow.set_tag(
            "task",
            "sentiment_classification",
        )

        mlflow.set_tag(
            "model_selection",
            "final_candidate",
        )

        mlflow.set_tag(
            "pipeline",
            "TF-IDF + Balanced Linear SVM",
        )

        mlflow.set_tag(
            "dataset",
            "synthetic_product_reviews",
        )

        # ----------------------------------------------------
        # Log and register complete pipeline
        # ----------------------------------------------------

        model_info = (
            mlflow.sklearn.log_model(
                sk_model=model,
                name="sentiment_pipeline",
                registered_model_name=(
                    REGISTERED_MODEL_NAME
                ),
            )
        )

        print()
        print(
            "Model logged successfully."
        )

        print(
            f"Model URI: "
            f"{model_info.model_uri}"
        )

        print()
        print(
            "Registered model:"
        )

        print(
            REGISTERED_MODEL_NAME
        )

    print()
    print(
        "=" * 70
    )

    print(
        "MODEL REGISTRATION COMPLETED"
    )

    print(
        "=" * 70
    )


# ============================================================
# Main
# ============================================================

def main():

    configure_mlflow()

    train_and_register()


if __name__ == "__main__":
    main()