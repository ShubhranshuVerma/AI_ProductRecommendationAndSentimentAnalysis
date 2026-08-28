from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw")


  # File Loading
  
def load_dataset(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    return pd.read_csv(path)


  # Basic Information
  
def print_basic_information(
    customers,
    products,
    interactions,
    reviews,
):
    print("=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)

    print(f"Customers:    {len(customers):,}")
    print(f"Products:     {len(products):,}")
    print(f"Interactions: {len(interactions):,}")
    print(f"Reviews:      {len(reviews):,}")

    print()


  # Missing Values
  
def check_missing_values(
    customers,
    products,
    interactions,
    reviews,
):
    print("=" * 70)
    print("MISSING VALUE CHECK")
    print("=" * 70)

    datasets = {
        "customers": customers,
        "products": products,
        "interactions": interactions,
        "reviews": reviews,
    }

    for name, dataframe in datasets.items():
        missing = dataframe.isna().sum()

        missing = missing[missing > 0]

        if missing.empty:
            print(f"{name}: No missing values")
        else:
            print(f"{name}:")
            print(missing)

    print()


  # Duplicate Checks
  
def check_duplicates(
    customers,
    products,
    interactions,
    reviews,
):
    print("=" * 70)
    print("DUPLICATE CHECK")
    print("=" * 70)

    datasets = {
        "customers": customers,
        "products": products,
        "interactions": interactions,
        "reviews": reviews,
    }

    id_columns = {
        "customers": "customer_id",
        "products": "product_id",
        "interactions": "interaction_id",
        "reviews": "review_id",
    }

    for name, dataframe in datasets.items():
        duplicate_rows = dataframe.duplicated().sum()

        id_column = id_columns[name]
        duplicate_ids = dataframe[id_column].duplicated().sum()

        print(
            f"{name}: "
            f"{duplicate_rows} duplicate rows, "
            f"{duplicate_ids} duplicate IDs"
        )
    duplicate_review_text = (
        reviews["review_text"]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate review texts:",
        duplicate_review_text,
    )

    print()


  # Referential Integrity
  
def check_referential_integrity(
    customers,
    products,
    interactions,
    reviews,
):
    print("=" * 70)
    print("REFERENTIAL INTEGRITY")
    print("=" * 70)

    customer_ids = set(
        customers["customer_id"]
    )

    product_ids = set(
        products["product_id"]
    )

    invalid_interaction_customers = (
        ~interactions["customer_id"].isin(customer_ids)
    ).sum()

    invalid_interaction_products = (
        ~interactions["product_id"].isin(product_ids)
    ).sum()

    invalid_review_customers = (
        ~reviews["customer_id"].isin(customer_ids)
    ).sum()

    invalid_review_products = (
        ~reviews["product_id"].isin(product_ids)
    ).sum()

    print(
        "Invalid interaction customer IDs:",
        invalid_interaction_customers,
    )

    print(
        "Invalid interaction product IDs:",
        invalid_interaction_products,
    )

    print(
        "Invalid review customer IDs:",
        invalid_review_customers,
    )

    print(
        "Invalid review product IDs:",
        invalid_review_products,
    )

    assert invalid_interaction_customers == 0
    assert invalid_interaction_products == 0
    assert invalid_review_customers == 0
    assert invalid_review_products == 0

    print("Referential integrity: PASSED")
    print()


  # Rating Analysis
  
def analyze_ratings(interactions, reviews):
    print("=" * 70)
    print("RATING ANALYSIS")
    print("=" * 70)

    print("Review ratings:")
    print(
        reviews["rating"]
        .value_counts()
        .sort_index()
    )

    print()

    print("Average review rating:")
    print(
        round(
            reviews["rating"].mean(),
            2,
        )
    )

    print()

    rating_interactions = interactions[
        interactions["interaction_type"] == "rating"
    ]

    print(
        "Explicit rating interactions:",
        len(rating_interactions),
    )

    print()


  # Sentiment Analysis
  
def analyze_sentiment(reviews):
    print("=" * 70)
    print("SENTIMENT ANALYSIS")
    print("=" * 70)

    counts = reviews["sentiment"].value_counts()

    percentages = (
        reviews["sentiment"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    sentiment_table = pd.DataFrame(
        {
            "count": counts,
            "percentage": percentages,
        }
    )

    print(sentiment_table)

    print()

    print("Sentiment by rating:")

    sentiment_rating = pd.crosstab(
        reviews["rating"],
        reviews["sentiment"],
    )

    print(sentiment_rating)

    print()


  # Interaction Analysis
  
def analyze_interactions(interactions):
    print("=" * 70)
    print("INTERACTION ANALYSIS")
    print("=" * 70)

    counts = interactions[
        "interaction_type"
    ].value_counts()

    percentages = (
        interactions[
            "interaction_type"
        ]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    interaction_table = pd.DataFrame(
        {
            "count": counts,
            "percentage": percentages,
        }
    )

    print(interaction_table)

    print()

    print("Average interaction score:")
    print(
        round(
            interactions[
                "interaction_score"
            ].mean(),
            3,
        )
    )

    print()


  # Customer Analysis
  
def analyze_customers(
    customers,
    interactions,
):
    print("=" * 70)
    print("CUSTOMER ANALYSIS")
    print("=" * 70)

    print("Preferred category distribution:")

    print(
        customers[
            "preferred_category"
        ]
        .value_counts()
    )

    print()

    customer_interaction_counts = (
        interactions.groupby(
            "customer_id"
        )
        .size()
    )

    print(
        "Average interactions per customer:",
        round(
            customer_interaction_counts.mean(),
            2,
        ),
    )

    print(
        "Customers with no interactions:",
        len(customers)
        - customer_interaction_counts.shape[0],
    )

    print()

    print(
        "Customers with at least one interaction:",
        customer_interaction_counts.shape[0],
    )

    print()


  # Recommendation Signal Analysis
  
def analyze_recommendation_signal(
    customers,
    products,
    interactions,
):
    print("=" * 70)
    print("RECOMMENDATION SIGNAL ANALYSIS")
    print("=" * 70)

    merged = interactions.merge(
        customers[
            [
                "customer_id",
                "preferred_category",
                "secondary_category",
            ]
        ],
        on="customer_id",
        how="left",
    )

    merged = merged.merge(
        products[
            [
                "product_id",
                "category",
            ]
        ],
        on="product_id",
        how="left",
    )

    merged["category_match"] = (
        merged["category"]
        == merged["preferred_category"]
    )

    merged["secondary_match"] = (
        merged["category"]
        == merged["secondary_category"]
    )

    primary_match_rate = (
        merged["category_match"].mean()
        * 100
    )

    secondary_match_rate = (
        merged["secondary_match"].mean()
        * 100
    )

    print(
        "Interactions matching primary category:",
        f"{primary_match_rate:.2f}%",
    )

    print(
        "Interactions matching secondary category:",
        f"{secondary_match_rate:.2f}%",
    )

    print()

    print(
        "Average interaction score for primary category:",
        round(
            merged.loc[
                merged["category_match"],
                "interaction_score",
            ].mean(),
            3,
        ),
    )

    print(
        "Average interaction score for other categories:",
        round(
            merged.loc[
                ~merged["category_match"],
                "interaction_score",
            ].mean(),
            3,
        ),
    )

    print()


  # Leakage Check
  
def check_possible_leakage(reviews):
    print("=" * 70)
    print("POTENTIAL DATA LEAKAGE CHECK")
    print("=" * 70)

    print(
        "Sentiment column exists:",
        "sentiment" in reviews.columns,
    )

    print(
        "Rating column exists:",
        "rating" in reviews.columns,
    )

    print()
    print(
        "Important: rating must NOT be used as a sentiment "
        "model feature."
    )

    print(
        "The sentiment model will use review_text as the "
        "primary input."
    )

    print()


  # Main Validation
  
def main():
    customers = load_dataset(
        "customers.csv"
    )

    products = load_dataset(
        "products.csv"
    )

    interactions = load_dataset(
        "interactions.csv"
    )

    reviews = load_dataset(
        "reviews.csv"
    )

    print_basic_information(
        customers,
        products,
        interactions,
        reviews,
    )

    check_missing_values(
        customers,
        products,
        interactions,
        reviews,
    )

    check_duplicates(
        customers,
        products,
        interactions,
        reviews,
    )

    check_referential_integrity(
        customers,
        products,
        interactions,
        reviews,
    )

    analyze_ratings(
        interactions,
        reviews,
    )

    analyze_sentiment(
        reviews,
    )

    analyze_interactions(
        interactions,
    )

    analyze_customers(
        customers,
        interactions,
    )

    analyze_recommendation_signal(
        customers,
        products,
        interactions,
    )

    check_possible_leakage(
        reviews,
    )

    print("=" * 70)
    print("DATASET EXPLORATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()