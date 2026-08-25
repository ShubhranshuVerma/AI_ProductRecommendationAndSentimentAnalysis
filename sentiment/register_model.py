import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
from sklearn.pipeline import FeatureUnion
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

DATA_DIR = Path("data/processed/sentiment")
EXPERIMENT_NAME = "AI_Product_Recommendation_Sentiment"
REGISTERED_MODEL_NAME = "ProductSentimentModel"
RANDOM_STATE = 42
UNSEEN_THRESHOLD = 0.75
UNSEEN_REVIEWS = [
    ("positive", "The battery lasted through a full day of work and I never had to worry about charging it."),
    ("positive", "Everything was straightforward from the moment I opened the package, and the product has performed consistently."),
    ("positive", "After several weeks of use, I am impressed by how dependable the product has been."),
    ("positive", "The controls make sense immediately and using the product feels effortless."),
    ("positive", "I use this regularly and it has handled everything I have asked it to do without problems."),
    ("positive", "The overall experience has been better than I expected and I have no major complaints."),
    ("positive", "The product feels well made and has remained dependable during repeated use."),
    ("positive", "I found the setup simple and the day-to-day operation has been very smooth."),
    ("positive", "Performance has remained stable even after using it for extended periods."),
    ("positive", "This turned out to be a worthwhile purchase because it consistently does what I need."),
    ("positive", "The product has made my routine easier and I am satisfied with its overall performance."),
    ("positive", "I have used several alternatives before, and this one has been noticeably more dependable."),

    ("neutral", "The product handles its basic purpose, although there is nothing particularly remarkable about it."),
    ("neutral", "It performs adequately for ordinary use, but I can see areas where it could be refined."),
    ("neutral", "My experience has been fairly typical, with some useful aspects and some minor limitations."),
    ("neutral", "The product does what it is supposed to do, although it does not stand out from similar options."),
    ("neutral", "I have not experienced any major problems, but I have not been especially impressed either."),
    ("neutral", "For normal situations the product is sufficient, though expectations should remain reasonable."),
    ("neutral", "The main functionality works, while a few smaller details could be improved."),
    ("neutral", "It has been usable so far, although the overall experience feels fairly ordinary."),
    ("neutral", "The product is neither particularly impressive nor disappointing for my needs."),
    ("neutral", "Some parts work better than others, making the overall experience fairly average."),
    ("neutral", "It meets the basic requirements, but I would not consider it exceptional."),
    ("neutral", "The product has been acceptable overall, with both advantages and compromises."),

    ("negative", "The product started causing problems after repeated use and I can no longer rely on it."),
    ("negative", "Several functions have behaved unpredictably, which has made the product frustrating to use."),
    ("negative", "The materials do not seem durable and I have already noticed visible wear."),
    ("negative", "Performance drops noticeably when the product is used for longer periods."),
    ("negative", "I encountered multiple problems during normal use that I did not expect from the product."),
    ("negative", "The product has been unreliable and has made the task more difficult rather than easier."),
    ("negative", "After using it regularly, I have become increasingly dissatisfied with its consistency."),
    ("negative", "The product looked promising initially, but its actual performance has fallen short."),
    ("negative", "Important parts of the product have not worked properly and this has affected the overall experience."),
    ("negative", "I have had to work around several issues that should not occur during ordinary use."),
    ("negative", "The product does not perform consistently enough for me to depend on it."),
    ("negative", "I expected a more dependable experience, but repeated problems have made me regret the purchase.")
]

def load_data():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    return train_df["review_text"], train_df["sentiment"], test_df["review_text"], test_df["sentiment"]

def create_models():
    def tfidf():
        return FeatureUnion([
            ("word", TfidfVectorizer(lowercase=True, analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)),
            ("char", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_df=0.98, sublinear_tf=True)),
        ])
    return {
        "logistic_regression": Pipeline([("tfidf", tfidf()), ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]),
        "linear_svm": Pipeline([("tfidf", tfidf()), ("classifier", LinearSVC(random_state=RANDOM_STATE))]),
        "balanced_linear_svm": Pipeline([("tfidf", tfidf()), ("classifier", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE))]),
        "naive_bayes": Pipeline([("tfidf", tfidf()), ("classifier", MultinomialNB())])
    }

def evaluate(model, X, y):
    pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, pred),
        "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "negative_f1": f1_score(y, pred, labels=["negative"], average="macro", zero_division=0),
        "negative_recall": recall_score(y, pred, labels=["negative"], average="macro", zero_division=0),
        "weighted_precision": precision_score(y, pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y, pred, average="weighted", zero_division=0)
    }

def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Loading sentiment dataset...")
    X_train, y_train, X_test, y_test = load_data()
    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples: {len(X_test):,}")

    models = create_models()
    results = []

    print("\n" + "=" * 70)
    print("MODEL SELECTION")
    print("=" * 70)

    for name, model in models.items():
        print(f"\nTraining: {name}")
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        metrics["model"] = name
        results.append(metrics)
        print(f"Accuracy: {metrics['accuracy']:.4f} | Weighted F1: {metrics['weighted_f1']:.4f} | Macro F1: {metrics['macro_f1']:.4f} | Negative Recall: {metrics['negative_recall']:.4f}")

    results_df = pd.DataFrame(results)
    results_df["selection_score"] = 0.5 * results_df["macro_f1"] + 0.5 * results_df["negative_recall"]
    results_df = results_df.sort_values("selection_score", ascending=False)

    selected_name = results_df.iloc[0]["model"]
    selected_model = models[selected_name]

    print("\n" + "=" * 70)
    print("SELECTED MODEL")
    print("=" * 70)
    print(results_df[["model", "accuracy", "weighted_f1", "macro_f1", "negative_recall", "selection_score"]].to_string(index=False))
    print(f"\nSelected: {selected_name}")

    unseen_df = pd.DataFrame(UNSEEN_REVIEWS, columns=["actual", "review_text"])
    print(f"Unseen samples: {len(unseen_df)}")
    print(unseen_df["actual"].value_counts().to_string())
    unseen_pred = selected_model.predict(unseen_df["review_text"])
    unseen_df["predicted"] = unseen_pred
    unseen_accuracy = accuracy_score(unseen_df["actual"], unseen_pred)

    print("\n" + "=" * 70)
    print("UNSEEN REVIEW EVALUATION")
    print("=" * 70)
    print(unseen_df[["actual", "predicted", "review_text"]].to_string(index=False))
    print(f"\nUnseen accuracy: {unseen_accuracy:.4f}")
    print(classification_report(unseen_df["actual"], unseen_pred, zero_division=0))

    if unseen_accuracy < UNSEEN_THRESHOLD:
        print(f"\nMODEL NOT REGISTERED: unseen accuracy {unseen_accuracy:.4f} < {UNSEEN_THRESHOLD:.2f}")
        return

    final_model = create_models()[selected_name]
    final_model.fit(X_train, y_train)
    test_metrics = evaluate(final_model, X_test, y_test)

    print("\n" + "=" * 70)
    print("MLFLOW REGISTRATION")
    print("=" * 70)

    with mlflow.start_run(run_name=f"final_{selected_name}_pipeline") as run:
        mlflow.log_param("model_type", selected_name)
        mlflow.log_param("selection_criterion", "0.5_macro_f1 + 0.5_negative_recall")
        mlflow.log_param("unseen_threshold", UNSEEN_THRESHOLD)
        mlflow.log_metric("unseen_accuracy", float(unseen_accuracy))
        for key, value in test_metrics.items(): mlflow.log_metric(key, float(value))
        mlflow.set_tag("task", "sentiment_classification")
        mlflow.set_tag("model_selection", "validated_candidate")
        mlflow.set_tag("dataset", "synthetic_product_reviews")
        mlflow.set_tag("unseen_validation", "passed")
        model_info = mlflow.sklearn.log_model(sk_model=final_model, name="sentiment_pipeline", registered_model_name=REGISTERED_MODEL_NAME)
        print(f"Model URI: {model_info.model_uri}")
        print(f"Registered model: {REGISTERED_MODEL_NAME}")

    print("\n" + "=" * 70)
    print("MODEL REGISTRATION COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    main()