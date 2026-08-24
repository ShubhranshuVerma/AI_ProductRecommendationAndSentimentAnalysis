from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)


ARTIFACT_DIR = Path(
    "artifacts/sentiment_models"
)

LABELS = [
    "negative",
    "neutral",
    "positive",
]


def calculate_metrics(
    y_true,
    y_pred,
):
    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "weighted_precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "weighted_recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }


def get_class_metrics(
    y_true,
    y_pred,
):
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    return {
        "negative_precision": report[
            "negative"
        ]["precision"],
        "negative_recall": report[
            "negative"
        ]["recall"],
        "negative_f1": report[
            "negative"
        ]["f1-score"],
        "neutral_f1": report[
            "neutral"
        ]["f1-score"],
        "positive_f1": report[
            "positive"
        ]["f1-score"],
    }


def evaluate_predictions(
    model_name,
    y_true,
    y_pred,
):
    metrics = calculate_metrics(
        y_true,
        y_pred,
    )

    class_metrics = get_class_metrics(
        y_true,
        y_pred,
    )

    metrics.update(
        class_metrics
    )

    print()
    print("=" * 70)
    print(
        f"{model_name.upper()} RESULTS"
    )
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
            y_true,
            y_pred,
            labels=LABELS,
            zero_division=0,
        )
    )

    return metrics


def save_confusion_matrix(
    model_name,
    y_true,
    y_pred,
):
    output_dir = (
        ARTIFACT_DIR / model_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=LABELS,
    )

    display.plot()

    plt.title(
        f"{model_name} - Confusion Matrix"
    )

    plt.tight_layout()

    output_path = (
        output_dir
        / "confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Confusion matrix saved to: "
        f"{output_path}"
    )