import sys
from pathlib import Path
from sklearn.pipeline import FeatureUnion

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import mlflow
import mlflow.sklearn

import time

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from evaluate import (
    evaluate_predictions,
    save_confusion_matrix,
)

from mlflow_tracking.tracking import configure_mlflow


DATA_DIR = Path(
    "data/processed/sentiment"
)

ARTIFACT_DIR = Path(
    "artifacts/sentiment_models"
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

    return train_df, test_df


# ============================================================
# TF-IDF
# ============================================================

def create_tfidf_features(X_train, X_test):
    vectorizer = FeatureUnion([
        ("word", TfidfVectorizer(lowercase=True, analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)),
        ("char", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_df=0.98, sublinear_tf=True)),
    ])
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return vectorizer, X_train_tfidf, X_test_tfidf


# ============================================================
# Candidate models
# ============================================================

def create_models():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),

        "linear_svm": LinearSVC(
            random_state=RANDOM_STATE,
        ),

        "balanced_linear_svm": LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "naive_bayes": MultinomialNB(),
    }


def log_mlflow_run(
    model_name,
    model,
    metrics,
    training_time,
):
    """
    Log model parameters, metrics,
    tags, evaluation artifacts,
    and the trained model to MLflow.
    """

    with mlflow.start_run(
        run_name=model_name
    ):

        mlflow.log_param("model",model_name,)
        mlflow.log_param("feature_method", "word_char_tfidf")
        mlflow.log_param("word_ngram_range", "(1, 2)")
        mlflow.log_param("char_ngram_range", "(3, 5)")
        mlflow.log_param("word_min_df", 2)
        mlflow.log_param("char_min_df", 2)

        if hasattr(
            model,
            "get_params",
        ):

            parameters = (
                model.get_params()
            )

            for name, value in parameters.items():

                try:
                    mlflow.log_param(
                        name,
                        value,
                    )
                except Exception:
                    pass

        for name, value in metrics.items():

            if (
                name != "model"
                and name
                != "training_time_seconds"
            ):

                mlflow.log_metric(
                    name,
                    float(value),
                )

        mlflow.log_metric(
            "training_time_seconds",
            float(training_time),
        )

        mlflow.set_tag(
            "task",
            "sentiment_classification",
        )

        mlflow.set_tag("feature_method", "word + character TF-IDF")

        mlflow.set_tag(
            "dataset",
            "synthetic_product_reviews",
        )

        mlflow.set_tag(
            "candidate_model",
            model_name,
        )

        confusion_matrix_path = (
            Path("artifacts")
            / "sentiment_models"
            / model_name
            / "confusion_matrix.png"
        )

        if confusion_matrix_path.exists():

            mlflow.log_artifact(
                str(confusion_matrix_path),
                artifact_path="evaluation",
            )

        mlflow.sklearn.log_model(
            model,
            name="model",
        )

        print(
            f"MLflow run logged: {model_name}"
        )


# ============================================================
# Train and evaluate
# ============================================================

def train_and_evaluate(
    models,
    X_train,
    X_test,
    y_train,
    y_test,
):
    all_results = []

    for model_name, model in models.items():

        print()
        print(
            "=" * 70
        )

        print(
            f"Training: {model_name}"
        )

        print(
            "=" * 70
        )

        start_time = time.perf_counter()

        model.fit(
            X_train,
            y_train,
        )

        training_time = (
            time.perf_counter()
            - start_time
        )

        predictions = model.predict(
            X_test
        )

        metrics = evaluate_predictions(
            model_name,
            y_test,
            predictions,
        )

        save_confusion_matrix(
            model_name,
            y_test,
            predictions,
        )


        log_mlflow_run(
            model_name,
            model,
            metrics,
            training_time,
        )


        metrics["model"] = model_name
        metrics["training_time_seconds"] = (
            training_time
        )

        all_results.append(
            metrics
        )

        print(
            f"Training time: "
            f"{training_time:.4f} seconds"
        )

    return models, all_results


# ============================================================
# Save comparison
# ============================================================

def save_comparison(results):
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by="weighted_f1",
        ascending=False,
    )

    output_path = (
        ARTIFACT_DIR
        / "model_comparison.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "MODEL COMPARISON"
    )

    print(
        "=" * 70
    )

    display_columns = [
        "model",
        "accuracy",
        "weighted_f1",
        "macro_f1",
        "negative_f1",
        "negative_recall",
        "training_time_seconds",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Comparison saved to: "
        f"{output_path}"
    )

    return results_df


# ============================================================
# Main
# ============================================================

def main():

    configure_mlflow()

    print(
        "Loading sentiment dataset..."
    )

    train_df, test_df = load_data()

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
        "Creating shared TF-IDF features..."
    )

    (
        vectorizer,
        X_train_tfidf,
        X_test_tfidf,
    ) = create_tfidf_features(
        X_train,
        X_test,
    )

    print(
        f"Training TF-IDF shape: "
        f"{X_train_tfidf.shape}"
    )

    print(
        f"Testing TF-IDF shape: "
        f"{X_test_tfidf.shape}"
    )

    models = create_models()

    models, results = (
        train_and_evaluate(
            models,
            X_train_tfidf,
            X_test_tfidf,
            y_train,
            y_test,
        )
    )

    results_df = save_comparison(
        results
    )

    print()
    print(
        "Candidate model training completed."
    )

    return results_df


if __name__ == "__main__":
    main()