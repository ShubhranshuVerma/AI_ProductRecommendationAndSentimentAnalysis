from pathlib import Path

import pandas as pd


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path(
    "data/processed/recommendation"
)


# ============================================================
# Loading
# ============================================================

def load_recommendation_data():
    customers = pd.read_csv(
        RAW_DATA_DIR / "customers.csv"
    )

    products = pd.read_csv(
        RAW_DATA_DIR / "products.csv"
    )

    interactions = pd.read_csv(
        RAW_DATA_DIR / "interactions.csv"
    )

    return customers, products, interactions


# ============================================================
# Customer preprocessing
# ============================================================

def preprocess_customers(customers):
    customers = customers.copy()

    customers["customer_id"] = (
        customers["customer_id"]
        .astype(str)
        .str.strip()
    )

    customers["preferred_category"] = (
        customers["preferred_category"]
        .astype(str)
        .str.strip()
    )

    customers["secondary_category"] = (
        customers["secondary_category"]
        .astype(str)
        .str.strip()
    )

    customers["price_preference"] = (
        customers["price_preference"]
        .astype(str)
        .str.strip()
    )

    return customers


# ============================================================
# Product preprocessing
# ============================================================

def preprocess_products(products):
    products = products.copy()

    products["product_id"] = (
        products["product_id"]
        .astype(str)
        .str.strip()
    )

    products["category"] = (
        products["category"]
        .astype(str)
        .str.strip()
    )

    products["subcategory"] = (
        products["subcategory"]
        .astype(str)
        .str.strip()
    )

    products["brand"] = (
        products["brand"]
        .astype(str)
        .str.strip()
    )

    products["price"] = pd.to_numeric(
        products["price"],
        errors="coerce",
    )

    products["average_rating"] = pd.to_numeric(
        products["average_rating"],
        errors="coerce",
    )

    products["popularity_score"] = pd.to_numeric(
        products["popularity_score"],
        errors="coerce",
    )

    return products


# ============================================================
# Interaction preprocessing
# ============================================================

def preprocess_interactions(interactions):
    interactions = interactions.copy()

    interactions["customer_id"] = (
        interactions["customer_id"]
        .astype(str)
        .str.strip()
    )

    interactions["product_id"] = (
        interactions["product_id"]
        .astype(str)
        .str.strip()
    )

    interactions["interaction_type"] = (
        interactions["interaction_type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    interactions["interaction_score"] = pd.to_numeric(
        interactions["interaction_score"],
        errors="coerce",
    )

    interactions["rating"] = pd.to_numeric(
        interactions["rating"],
        errors="coerce",
    )

    interactions["timestamp"] = pd.to_datetime(
        interactions["timestamp"],
        errors="coerce",
    )

    interactions["quantity"] = pd.to_numeric(
        interactions["quantity"],
        errors="coerce",
    )

    return interactions


# ============================================================
# Interaction feature creation
# ============================================================

def create_interaction_features(interactions):
    interactions = interactions.copy()

    # Explicit rating is only available for rating interactions.
    # Missing values for other interaction types are expected.
    interactions["rating_signal"] = (
        interactions["rating"]
        .fillna(0)
        / 5.0
    )

    interactions["purchase_signal"] = (
        interactions["interaction_type"]
        == "purchase"
    ).astype(int)

    interactions["cart_signal"] = (
        interactions["interaction_type"]
        == "cart"
    ).astype(int)

    interactions["wishlist_signal"] = (
        interactions["interaction_type"]
        == "wishlist"
    ).astype(int)

    interactions["view_signal"] = (
        interactions["interaction_type"]
        == "view"
    ).astype(int)

    interactions["interaction_strength"] = (
        interactions["interaction_score"]
        + (
            interactions["rating_signal"]
            * 0.20
        )
    )

    return interactions


# ============================================================
# Referential validation
# ============================================================

def validate_recommendation_relationships(
    customers,
    products,
    interactions,
):
    valid_customers = set(
        customers["customer_id"]
    )

    valid_products = set(
        products["product_id"]
    )

    invalid_customers = (
        ~interactions["customer_id"].isin(
            valid_customers
        )
    ).sum()

    invalid_products = (
        ~interactions["product_id"].isin(
            valid_products
        )
    ).sum()

    if invalid_customers > 0:
        raise ValueError(
            f"Found {invalid_customers} invalid "
            "customer IDs in interactions."
        )

    if invalid_products > 0:
        raise ValueError(
            f"Found {invalid_products} invalid "
            "product IDs in interactions."
        )


# ============================================================
# Save
# ============================================================

def save_recommendation_data(
    customers,
    products,
    interactions,
):
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    customers.to_csv(
        PROCESSED_DATA_DIR / "customers.csv",
        index=False,
    )

    products.to_csv(
        PROCESSED_DATA_DIR / "products.csv",
        index=False,
    )

    interactions.to_csv(
        PROCESSED_DATA_DIR / "interactions.csv",
        index=False,
    )


# ============================================================
# Main pipeline
# ============================================================

def run_recommendation_preprocessing():
    print(
        "Loading recommendation data..."
    )

    (
        customers,
        products,
        interactions,
    ) = load_recommendation_data()

    print(
        "Preprocessing customers..."
    )

    customers = preprocess_customers(
        customers
    )

    print(
        "Preprocessing products..."
    )

    products = preprocess_products(
        products
    )

    print(
        "Preprocessing interactions..."
    )

    interactions = preprocess_interactions(
        interactions
    )

    print(
        "Creating interaction features..."
    )

    interactions = create_interaction_features(
        interactions
    )

    print(
        "Validating relationships..."
    )

    validate_recommendation_relationships(
        customers,
        products,
        interactions,
    )

    save_recommendation_data(
        customers,
        products,
        interactions,
    )

    print()
    print(
        "Recommendation preprocessing completed."
    )

    print(
        f"Customers: {len(customers):,}"
    )

    print(
        f"Products: {len(products):,}"
    )

    print(
        f"Interactions: {len(interactions):,}"
    )


if __name__ == "__main__":
    run_recommendation_preprocessing()