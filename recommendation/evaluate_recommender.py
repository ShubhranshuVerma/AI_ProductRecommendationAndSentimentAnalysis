from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from recommendation.recommender import ProductRecommender

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "recommendation"
RANDOM_STATE = 42
TEST_SIZE = 0.20
RELEVANT_TYPES = {"purchase", "rating"}


def split_interactions(interactions):
    train_parts = []
    test_parts = []

    for customer_id, group in interactions.groupby("customer_id"):
        group = group.sample(frac=1, random_state=RANDOM_STATE)

        if len(group) < 2:
            train_parts.append(group)
            continue

        test_count = max(1, int(len(group) * TEST_SIZE))
        test_count = min(test_count, len(group) - 1)

        test_parts.append(group.iloc[:test_count])
        train_parts.append(group.iloc[test_count:])

    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


def precision_at_k(recommended, relevant, k):
    recommended = recommended[:k]
    if not recommended:
        return 0.0
    return len(set(recommended) & set(relevant)) / k


def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0
    recommended = recommended[:k]
    return len(set(recommended) & set(relevant)) / len(set(relevant))


def hit_rate_at_k(recommended, relevant, k):
    recommended = recommended[:k]
    return float(bool(set(recommended) & set(relevant)))


def evaluate_recommender(recommender, test_df, k=10):
    results = []
    recommended_products = set()

    for customer_id, group in test_df.groupby("customer_id"):
        relevant = set(
            group[group["interaction_type"].isin(RELEVANT_TYPES)]["product_id"]
        )

        if not relevant:
            continue

        try:
            recommendations = recommender.recommend(customer_id, top_n=k)
        except ValueError:
            continue

        predicted = recommendations["product_id"].tolist()
        recommended_products.update(predicted)

        results.append({
            "customer_id": customer_id,
            "precision_at_10": precision_at_k(predicted, relevant, k),
            "recall_at_10": recall_at_k(predicted, relevant, k),
            "hit_rate_at_10": hit_rate_at_k(predicted, relevant, k),
        })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        raise RuntimeError("No customers had relevant test interactions.")

    metrics = {
        "precision_at_10": results_df["precision_at_10"].mean(),
        "recall_at_10": results_df["recall_at_10"].mean(),
        "hit_rate_at_10": results_df["hit_rate_at_10"].mean(),
        "coverage": len(recommended_products) / recommender.products["product_id"].nunique(),
    }

    return metrics


def create_popularity_recommender(train_df, products):
    popularity = (
        train_df.groupby("product_id")
        .size()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    popularity_rank = {product_id: i for i, product_id in enumerate(popularity)}

    def recommend(customer_id, top_n=10):
        history = set(
            train_df[train_df["customer_id"] == customer_id]["product_id"]
        )

        candidates = [
            product_id
            for product_id in popularity
            if product_id not in history
        ][:top_n]

        return pd.DataFrame({"product_id": candidates})

    return recommend


def evaluate_popularity(popularity_recommender, test_df, k=10):
    results = []
    recommended_products = set()

    for customer_id, group in test_df.groupby("customer_id"):
        relevant = set(
            group[group["interaction_type"].isin(RELEVANT_TYPES)]["product_id"]
        )

        if not relevant:
            continue

        recommendations = popularity_recommender(customer_id, k)
        predicted = recommendations["product_id"].tolist()
        recommended_products.update(predicted)

        results.append({
            "precision": precision_at_k(predicted, relevant, k),
            "recall": recall_at_k(predicted, relevant, k),
            "hit_rate": hit_rate_at_k(predicted, relevant, k),
        })

    return {
        "precision_at_10": pd.DataFrame(results)["precision"].mean(),
        "recall_at_10": pd.DataFrame(results)["recall"].mean(),
        "hit_rate_at_10": pd.DataFrame(results)["hit_rate"].mean(),
        "coverage": len(recommended_products) / 250,
    }


def main():
    print("=" * 70)
    print("RECOMMENDATION EVALUATION")
    print("=" * 70)

    interactions = pd.read_csv(DATA_DIR / "interactions.csv")
    products = pd.read_csv(DATA_DIR / "products.csv")

    train_df, test_df = split_interactions(interactions)

    print(f"Total interactions: {len(interactions):,}")
    print(f"Training interactions: {len(train_df):,}")
    print(f"Testing interactions: {len(test_df):,}")

    relevant_count = test_df[
        test_df["interaction_type"].isin(RELEVANT_TYPES)
    ].shape[0]

    print(f"Relevant test interactions: {relevant_count:,}")

    # Hybrid recommender
    recommender = ProductRecommender()
    recommender.interactions = train_df.copy()
    recommender._prepare_features()

    hybrid_metrics = evaluate_recommender(
        recommender,
        test_df,
    )

    # Popularity baseline
    popularity_recommender = create_popularity_recommender(
        train_df,
        products,
    )

    popularity_metrics = evaluate_popularity(
        popularity_recommender,
        test_df,
    )

    comparison = pd.DataFrame([
        {
            "model": "hybrid_recommender",
            **hybrid_metrics,
        },
        {
            "model": "popularity_baseline",
            **popularity_metrics,
        },
    ])

    print()
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(comparison.to_string(index=False))

    output_dir = PROJECT_ROOT / "artifacts" / "recommendation"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "recommendation_comparison.csv"
    comparison.to_csv(output_path, index=False)

    print()
    print(f"Results saved to: {output_path}")
    print("Recommendation evaluation completed.")


if __name__ == "__main__":
    main()