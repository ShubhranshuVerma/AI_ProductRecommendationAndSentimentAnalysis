from pathlib import Path
import sys
import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentiment.evaluate import evaluate_predictions

MODEL_PATH = Path("models/sentiment_model.joblib")

REVIEWS = [
    ("positive", "The battery easily lasted through my entire workday and the screen is remarkably bright."),
    ("positive", "Setup took only a few minutes and everything has worked smoothly since then."),
    ("positive", "I expected an average experience, but the product feels much more refined than I anticipated."),
    ("positive", "The controls are intuitive and the overall experience has been genuinely enjoyable."),
    ("negative", "After barely two weeks of use, the device began overheating and shutting down unexpectedly."),
    ("negative", "The materials feel cheap and several components have already started showing signs of wear."),
    ("negative", "Performance becomes inconsistent whenever I use the product for longer periods."),
    ("negative", "I regret buying this because it failed to meet even my basic expectations."),
    ("neutral", "The product performs its main function adequately, although I did not find anything particularly impressive."),
    ("neutral", "It works as advertised, but there are a few limitations that keep it from standing out."),
    ("neutral", "The overall experience is acceptable for the price, with both strengths and weaknesses."),
    ("neutral", "Nothing is seriously wrong with it, but I would not describe the experience as exceptional.")
]

def main():
    df = pd.DataFrame(REVIEWS, columns=["actual", "review_text"])
    model = joblib.load(MODEL_PATH)
    predictions = model.predict(df["review_text"])
    df["predicted"] = predictions
    print("=" * 70)
    print("UNSEEN REVIEW EVALUATION")
    print("=" * 70)
    print(df[["actual", "predicted", "review_text"]].to_string(index=False))
    print()
    evaluate_predictions("unseen_reviews", df["actual"], predictions)

if __name__ == "__main__":
    main()