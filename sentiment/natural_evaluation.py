from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


NATURAL_EVAL_PATH = (
    Path(__file__).resolve().parent
    / "natural_eval.csv"
)


def load_natural_evaluation():
    """
    Load manually curated natural-language reviews.

    These reviews are intentionally different from the
    synthetic review templates used during training.
    """

    return pd.read_csv(
        NATURAL_EVAL_PATH
    )


def evaluate_natural_language(
    model,
    vectorizer,
):
    """
    Evaluate a trained model on genuinely unseen,
    manually curated natural-language reviews.
    """

    df = load_natural_evaluation()

    X = vectorizer.transform(
        df["review_text"]
    )

    predictions = model.predict(X)

    accuracy = accuracy_score(
        df["sentiment"],
        predictions,
    )

    macro_f1 = f1_score(
        df["sentiment"],
        predictions,
        average="macro",
    )

    weighted_f1 = f1_score(
        df["sentiment"],
        predictions,
        average="weighted",
    )

    return {
        "natural_eval_accuracy": float(
            accuracy
        ),
        "natural_eval_macro_f1": float(
            macro_f1
        ),
        "natural_eval_weighted_f1": float(
            weighted_f1
        ),
    }
