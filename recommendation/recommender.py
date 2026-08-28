from pathlib import Path

import pandas as pd


  # Configuration
  
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recommendation"
)


INTERACTION_WEIGHTS = {
    "view": 1.0,
    "wishlist": 2.0,
    "cart": 3.0,
    "rating": 4.0,
    "purchase": 5.0,
}


  # Recommendation Engine
  
class ProductRecommender:

    def __init__(
        self,
        customers_path=None,
        products_path=None,
        interactions_path=None,
    ):

        self.customers_path = (
            customers_path
            or DATA_DIR / "customers.csv"
        )

        self.products_path = (
            products_path
            or DATA_DIR / "products.csv"
        )

        self.interactions_path = (
            interactions_path
            or DATA_DIR / "interactions.csv"
        )

        self.customers = None
        self.products = None
        self.interactions = None

        self.customer_category_scores = None
        self.product_popularity = None
        self.product_ratings = None

        self._load_data()
        self._prepare_features()

      
    # Load data
      

    def _load_data(self):

        self.customers = pd.read_csv(
            self.customers_path
        )

        self.products = pd.read_csv(
            self.products_path
        )

        self.interactions = pd.read_csv(
            self.interactions_path
        )

      
    # Prepare recommendation features
      

    def _prepare_features(self):

        interactions = self.interactions.copy()

          
        # Validate required columns
          

        required_interaction_columns = [
            "customer_id",
            "product_id",
            "interaction_type",
        ]

        missing_columns = [
            column
            for column in required_interaction_columns
            if column not in interactions.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing interaction columns: "
                + ", ".join(missing_columns)
            )

        required_product_columns = [
            "product_id",
            "category",
        ]

        missing_product_columns = [
            column
            for column in required_product_columns
            if column not in self.products.columns
        ]

        if missing_product_columns:
            raise ValueError(
                "Missing product columns: "
                + ", ".join(missing_product_columns)
            )

          
        # Add product category to interactions
          

        product_categories = (
            self.products[
                [
                    "product_id",
                    "category",
                ]
            ]
            .drop_duplicates(
                subset=["product_id"]
            )
        )

        interactions = interactions.merge(
            product_categories,
            on="product_id",
            how="left",
        )

        missing_categories = (
            interactions["category"]
            .isna()
            .sum()
        )

        if missing_categories > 0:
            raise ValueError(
                f"{missing_categories} interactions "
                "could not be matched to a product category."
            )

          
        # Interaction strength
          

        interactions["interaction_weight"] = (
            interactions[
                "interaction_type"
            ]
            .map(
                INTERACTION_WEIGHTS
            )
            .fillna(0)
        )

          
        # Customer-category preference
          

        category_scores = (
            interactions
            .groupby(
                [
                    "customer_id",
                    "category",
                ]
            )["interaction_weight"]
            .sum()
            .reset_index()
        )

        category_totals = (
            category_scores
            .groupby(
                "customer_id"
            )["interaction_weight"]
            .transform(
                "sum"
            )
        )

        category_scores[
            "category_score"
        ] = (
            category_scores[
                "interaction_weight"
            ]
            / category_totals.replace(
                0,
                1,
            )
        )

        self.customer_category_scores = (
            category_scores
        )

          
        # Product popularity
          

        popularity = (
            interactions
            .groupby(
                "product_id"
            )["interaction_weight"]
            .sum()
        )

        max_popularity = (
            popularity.max()
        )

        if max_popularity > 0:

            popularity = (
                popularity
                / max_popularity
            )

        self.product_popularity = (
            popularity
        )

          
        # Product ratings
          

        if "rating" in interactions.columns:

            ratings = (
                interactions[
                    interactions[
                        "rating"
                    ].notna()
                ]
                .groupby(
                    "product_id"
                )["rating"]
                .mean()
            )

            self.product_ratings = (
                ratings
            )

        else:

            self.product_ratings = (
                pd.Series(
                    dtype=float
                )
            )

      
    # Get customer history
      

    def _get_customer_history(
        self,
        customer_id,
    ):

        return self.interactions[
            self.interactions[
                "customer_id"
            ]
            == customer_id
        ]

      
    # Score products
      

    def _score_products(
        self,
        customer_id,
    ):

        history = (
            self._get_customer_history(
                customer_id
            )
        )

        customer_categories = (
            self.customer_category_scores[
                self.customer_category_scores[
                    "customer_id"
                ]
                == customer_id
            ]
        )

        category_map = dict(
            zip(
                customer_categories[
                    "category"
                ],
                customer_categories[
                    "category_score"
                ],
            )
        )

        products = (
            self.products.copy()
        )

          
        # Category affinity
          

        products[
            "category_affinity"
        ] = products[
            "category"
        ].map(
            category_map
        ).fillna(0)

          
        # Popularity
          

        products[
            "popularity_score"
        ] = products[
            "product_id"
        ].map(
            self.product_popularity
        ).fillna(0)

          
        # Rating
          

        products[
            "rating_score"
        ] = products[
            "product_id"
        ].map(
            self.product_ratings
        ).fillna(0)

        products[
            "rating_score"
        ] = (
            products[
                "rating_score"
            ]
            / 5.0
        )

          
        # Combined recommendation score
          

        products[
            "recommendation_score"
        ] = (
            0.60
            * products[
                "category_affinity"
            ]
            +
            0.25
            * products[
                "popularity_score"
            ]
            +
            0.15
            * products[
                "rating_score"
            ]
        )

          
        # Remove products already seen
          

        seen_products = set(
            history[
                "product_id"
            ]
        )

        products = products[
            ~products[
                "product_id"
            ].isin(
                seen_products
            )
        ]

        return products

      
    # Recommend
      

    def recommend(
        self,
        customer_id,
        top_n=10,
    ):

        if top_n <= 0:
            raise ValueError(
                "top_n must be greater than 0."
            )

        known_customers = set(
            self.customers[
                "customer_id"
            ]
        )

        if customer_id not in known_customers:
            raise ValueError(
                f"Unknown customer_id: "
                f"{customer_id}"
            )

        history = (
            self._get_customer_history(
                customer_id
            )
        )

          
        # Existing customer
          

        if not history.empty:

            recommendations = (
                self._score_products(
                    customer_id
                )
            )

          
        # Cold-start customer
          

        else:

            recommendations = (
                self.products.copy()
            )

            recommendations[
                "popularity_score"
            ] = recommendations[
                "product_id"
            ].map(
                self.product_popularity
            ).fillna(0)

            recommendations[
                "rating_score"
            ] = recommendations[
                "product_id"
            ].map(
                self.product_ratings
            ).fillna(0)

            recommendations[
                "rating_score"
            ] = (
                recommendations[
                    "rating_score"
                ]
                / 5.0
            )

            recommendations[
                "recommendation_score"
            ] = (
                0.70
                * recommendations[
                    "popularity_score"
                ]
                +
                0.30
                * recommendations[
                    "rating_score"
                ]
            )

          
        # Sort and select Top-N
          

        recommendations = (
            recommendations
            .sort_values(
                by=[
                    "recommendation_score",
                    "product_id",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .head(
                top_n
            )
        )

        output_columns = [
            "product_id",
            "product_name",
            "category",
            "recommendation_score",
        ]

        available_columns = [
            column
            for column in output_columns
            if column in recommendations.columns
        ]

        return (
            recommendations[
                available_columns
            ]
            .reset_index(
                drop=True
            )
        )