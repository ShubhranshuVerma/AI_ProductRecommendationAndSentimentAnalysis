import sys
from pathlib import Path

# ------------------------------------------------------------
# Add project root to Python path
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import mlflow


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "ProductSentimentModel"
MODEL_VERSION = "1"

TRACKING_URI = "sqlite:///mlflow.db"


# ============================================================
# Load registered model
# ============================================================

def load_registered_model():

    print("=" * 70)
    print("REGISTERED MODEL VERIFICATION")
    print("=" * 70)

    print()
    print("Connecting to MLflow...")

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    model_uri = (
        f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    )

    print(
        f"Loading model: {model_uri}"
    )

    model = mlflow.pyfunc.load_model(
        model_uri
    )

    print()
    print("Model loaded successfully.")

    return model


# ============================================================
# Test predictions
# ============================================================

def test_predictions(model):

    test_reviews = [
        "The product quality is excellent and I am very happy with it.",
        "The product is okay, nothing special.",
        "Very poor quality and completely disappointing.",
    ]

    print()
    print("=" * 70)
    print("TEST PREDICTIONS")
    print("=" * 70)

    predictions = model.predict(
        test_reviews
    )

    for review, prediction in zip(
        test_reviews,
        predictions,
    ):

        print()
        print(f"Review: {review}")
        print(f"Prediction: {prediction}")


# ============================================================
# Main
# ============================================================

def main():

    model = load_registered_model()

    test_predictions(
        model
    )

    print()
    print("=" * 70)
    print("REGISTERED MODEL VERIFICATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()