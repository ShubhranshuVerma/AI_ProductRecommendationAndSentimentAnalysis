import sys
from pathlib import Path

# ------------------------------------------------------------
# Add project root to Python path
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import mlflow
import pandas as pd

from review_analysis import analyze_reviews


# ============================================================
# Configuration
# ============================================================

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "reviews.csv"
)

MODEL_NAME = "ProductSentimentModel"
MODEL_VERSION = "1"

TRACKING_URI = "sqlite:///mlflow.db"

RANDOM_STATE = 42

# Number of reviews supplied to the LLM
REVIEWS_PER_SENTIMENT = 5


# ============================================================
# Load registered model
# ============================================================

def load_registered_model():

    print()
    print(
        "Loading registered sentiment model..."
    )

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    model_uri = (
        f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    )

    model = mlflow.pyfunc.load_model(
        model_uri
    )

    print(
        "Registered model loaded:"
    )

    print(
        f"{MODEL_NAME} "
        f"Version {MODEL_VERSION}"
    )

    return model


# ============================================================
# Load reviews
# ============================================================

def load_reviews():

    print()
    print(
        "Loading reviews..."
    )

    reviews_df = pd.read_csv(
        DATA_PATH
    )

    required_columns = [
        "review_text",
        "product_id",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in reviews_df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required review columns: "
            + ", ".join(
                missing_columns
            )
        )

    reviews_df = reviews_df.dropna(
        subset=["review_text"]
    ).copy()

    reviews_df["review_text"] = (
        reviews_df["review_text"]
        .astype(str)
        .str.strip()
    )

    reviews_df = reviews_df[
        reviews_df["review_text"] != ""
    ]

    print(
        f"Reviews loaded: "
        f"{len(reviews_df):,}"
    )

    return reviews_df


# ============================================================
# Predict sentiments
# ============================================================

def predict_sentiments(
    model,
    reviews_df,
):

    print()
    print(
        "Generating sentiment predictions..."
    )

    predictions = model.predict(
        reviews_df["review_text"].tolist()
    )

    reviews_df = reviews_df.copy()

    reviews_df["predicted_sentiment"] = (
        predictions
    )

    print()
    print(
        "Predicted sentiment distribution:"
    )

    print(
        reviews_df[
            "predicted_sentiment"
        ]
        .value_counts()
        .to_string()
    )

    return reviews_df


# ============================================================
# Select representative reviews
# ============================================================

def select_representative_reviews(
    reviews_df,
):

    print()
    print(
        "Selecting representative reviews..."
    )

    selected_frames = []

    for sentiment in [
        "negative",
        "neutral",
        "positive",
    ]:

        sentiment_df = reviews_df[
            reviews_df[
                "predicted_sentiment"
            ]
            == sentiment
        ]

        if sentiment_df.empty:
            continue

        sample_size = min(
            REVIEWS_PER_SENTIMENT,
            len(sentiment_df),
        )

        sampled = sentiment_df.sample(
            n=sample_size,
            random_state=RANDOM_STATE,
        )

        selected_frames.append(
            sampled
        )

    if not selected_frames:

        raise ValueError(
            "No reviews available "
            "for LLM analysis."
        )

    selected_df = pd.concat(
        selected_frames,
        ignore_index=True,
    )

    print(
        f"Selected reviews: "
        f"{len(selected_df)}"
    )

    print()
    print(
        "Selected sentiment distribution:"
    )

    print(
        selected_df[
            "predicted_sentiment"
        ]
        .value_counts()
        .to_string()
    )

    return selected_df


# ============================================================
# Prepare LLM input
# ============================================================

def prepare_llm_reviews(
    reviews_df,
):

    reviews = []

    for _, row in reviews_df.iterrows():

        reviews.append(
            {
                "sentiment": row[
                    "predicted_sentiment"
                ],
                "review_text": row[
                    "review_text"
                ],
            }
        )

    return reviews


# ============================================================
# Run LLM analysis
# ============================================================

def run_llm_analysis(
    reviews,
):

    print()
    print(
        "Sending representative reviews "
        "to Gemini..."
    )

    result = analyze_reviews(
        product_name=(
            "Product catalog feedback"
        ),
        reviews=reviews,
    )

    return result


# ============================================================
# Display results
# ============================================================

def display_results(
    result,
):

    print()
    print(
        "=" * 70
    )

    print(
        "LLM PRODUCT REVIEW ANALYSIS"
    )

    print(
        "=" * 70
    )

    print()
    print(
        "Summary:"
    )

    print(
        result["summary"]
    )

    print()
    print(
        "Common Complaints:"
    )

    for item in result[
        "common_complaints"
    ]:

        print(
            f"- {item}"
        )

    print()
    print(
        "Praised Features:"
    )

    for item in result[
        "praised_features"
    ]:

        print(
            f"- {item}"
        )

    print()
    print(
        "Business Insights:"
    )

    for item in result[
        "business_insights"
    ]:

        print(
            f"- {item}"
        )


# ============================================================
# Main
# ============================================================

def main():

    model = load_registered_model()

    reviews_df = load_reviews()

    reviews_df = predict_sentiments(
        model,
        reviews_df,
    )

    selected_df = (
        select_representative_reviews(
            reviews_df
        )
    )

    reviews = prepare_llm_reviews(
        selected_df
    )

    result = run_llm_analysis(
        reviews
    )

    display_results(
        result
    )

    print()
    print(
        "=" * 70
    )

    print(
        "ML → LLM INTEGRATION COMPLETED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()