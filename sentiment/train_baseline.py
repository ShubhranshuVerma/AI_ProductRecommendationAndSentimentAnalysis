from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path(
    "data/processed/sentiment"
)

ARTIFACT_DIR = Path(
    "artifacts/sentiment_baseline"
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

def create_tfidf_features(
    X_train,
    X_test,
):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(
        X_train
    )

    X_test_tfidf = vectorizer.transform(
        X_test
    )

    return (
        vectorizer,
        X_train_tfidf,
        X_test_tfidf,
    )


# ============================================================
# Train model
# ============================================================

def train_model(
    X_train,
    y_train,
):
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
):
    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print()
    print("=" * 70)
    print("BASELINE MODEL RESULTS")
    print("=" * 70)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
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

    return predictions


# ============================================================
# Confusion matrix
# ============================================================

def save_confusion_matrix(
    y_test,
    predictions,
):
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[
            "negative",
            "neutral",
            "positive",
        ],
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "negative",
            "neutral",
            "positive",
        ],
    )

    display.plot()

    plt.title(
        "Sentiment Baseline - Confusion Matrix"
    )

    plt.tight_layout()

    output_path = (
        ARTIFACT_DIR
        / "confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Confusion matrix saved to: {output_path}"
    )


# ============================================================
# Main
# ============================================================

def main():
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
        f"Training samples: {len(X_train):,}"
    )

    print(
        f"Testing samples: {len(X_test):,}"
    )

    print()
    print(
        "Creating TF-IDF features..."
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

    print()
    print(
        "Training Logistic Regression..."
    )

    model = train_model(
        X_train_tfidf,
        y_train,
    )

    predictions = evaluate_model(
        model,
        X_test_tfidf,
        y_test,
    )

    save_confusion_matrix(
        y_test,
        predictions,
    )

    print()
    print("=" * 70)
    print("BASELINE TRAINING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()