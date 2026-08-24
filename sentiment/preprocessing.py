from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path(
    "data/processed/sentiment"
)

RANDOM_STATE = 42


# ============================================================
# Loading
# ============================================================

def load_reviews():
    return pd.read_csv(
        RAW_DATA_DIR / "reviews.csv"
    )


# ============================================================
# Text cleaning
# ============================================================

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Normalize whitespace.
    text = " ".join(
        text.split()
    )

    return text.strip()


def preprocess_reviews(reviews):
    reviews = reviews.copy()

    reviews["review_text"] = (
        reviews["review_text"]
        .apply(clean_text)
    )

    reviews["review_title"] = (
        reviews["review_title"]
        .apply(clean_text)
    )

    reviews["sentiment"] = (
        reviews["sentiment"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return reviews


# ============================================================
# Validation
# ============================================================

def validate_reviews(reviews):
    valid_sentiments = {
        "positive",
        "neutral",
        "negative",
    }

    invalid_sentiments = set(
        reviews["sentiment"]
    ) - valid_sentiments

    if invalid_sentiments:
        raise ValueError(
            "Invalid sentiment labels found: "
            f"{invalid_sentiments}"
        )

    empty_reviews = (
        reviews["review_text"]
        .str.strip()
        .eq("")
        .sum()
    )

    if empty_reviews > 0:
        raise ValueError(
            f"Found {empty_reviews} empty reviews."
        )


# ============================================================
# Remove exact duplicate reviews
# ============================================================

def remove_duplicate_reviews(reviews):
    before = len(reviews)

    reviews = reviews.drop_duplicates(
        subset=["review_text"]
    ).copy()

    removed = (
        before - len(reviews)
    )

    print(
        f"Removed duplicate review texts: {removed}"
    )

    return reviews


# ============================================================
# Train/test split
# ============================================================

def split_reviews(reviews):
    X = reviews["review_text"]
    y = reviews["sentiment"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# Save
# ============================================================

def save_sentiment_data(
    X_train,
    X_test,
    y_train,
    y_test,
):
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df = pd.DataFrame(
        {
            "review_text": X_train,
            "sentiment": y_train,
        }
    )

    test_df = pd.DataFrame(
        {
            "review_text": X_test,
            "sentiment": y_test,
        }
    )

    train_df.to_csv(
        PROCESSED_DATA_DIR / "train.csv",
        index=False,
    )

    test_df.to_csv(
        PROCESSED_DATA_DIR / "test.csv",
        index=False,
    )


# ============================================================
# Main pipeline
# ============================================================

def run_sentiment_preprocessing():
    print(
        "Loading reviews..."
    )

    reviews = load_reviews()

    print(
        "Cleaning reviews..."
    )

    reviews = preprocess_reviews(
        reviews
    )

    print(
        "Validating reviews..."
    )

    validate_reviews(
        reviews
    )

    print(
        "Removing duplicate reviews..."
    )

    reviews = remove_duplicate_reviews(
        reviews
    )

    print(
        "Creating train/test split..."
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_reviews(
        reviews
    )

    save_sentiment_data(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print()
    print(
        "Sentiment preprocessing completed."
    )

    print(
        f"Training samples: {len(X_train):,}"
    )

    print(
        f"Testing samples: {len(X_test):,}"
    )

    print()
    print(
        "Training sentiment distribution:"
    )

    print(
        y_train.value_counts(
            normalize=True
        )
        .mul(100)
        .round(2)
    )

    print()
    print(
        "Testing sentiment distribution:"
    )

    print(
        y_test.value_counts(
            normalize=True
        )
        .mul(100)
        .round(2)
    )


if __name__ == "__main__":
    run_sentiment_preprocessing()