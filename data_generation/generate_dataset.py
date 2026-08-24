import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

SEED = 42

NUM_CUSTOMERS = 1000
NUM_PRODUCTS = 250
NUM_INTERACTIONS = 20000
NUM_REVIEWS = 5000

OUTPUT_DIR = Path("data/raw")

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# Product configuration
# ============================================================

CATEGORY_CONFIG = {
    "Electronics": {
        "subcategories": [
            "Audio",
            "Mobile",
            "Laptop",
            "Accessories",
            "Smart Home",
        ],
        "brands": [
            "SoundMax",
            "TechPro",
            "NovaTech",
            "Zenith",
            "Electra",
        ],
    },
    "Fashion": {
        "subcategories": [
            "Men",
            "Women",
            "Footwear",
            "Accessories",
            "Sportswear",
        ],
        "brands": [
            "UrbanStyle",
            "TrendX",
            "ClassicWear",
            "ModeFit",
            "StyleHub",
        ],
    },
    "Home & Kitchen": {
        "subcategories": [
            "Kitchen",
            "Furniture",
            "Decor",
            "Appliances",
            "Storage",
        ],
        "brands": [
            "HomeCraft",
            "KitchenPro",
            "ComfortLiving",
            "Casa",
            "DailyHome",
        ],
    },
    "Sports": {
        "subcategories": [
            "Fitness",
            "Running",
            "Outdoor",
            "Cricket",
            "Yoga",
        ],
        "brands": [
            "FitPro",
            "ActiveX",
            "SportMax",
            "Peak",
            "MoveWell",
        ],
    },
    "Books": {
        "subcategories": [
            "Fiction",
            "Technology",
            "Business",
            "Science",
            "Self Help",
        ],
        "brands": [
            "ReadWell",
            "BookHouse",
            "KnowledgeHub",
            "PaperTrail",
            "PageTurner",
        ],
    },
    "Beauty": {
        "subcategories": [
            "Skincare",
            "Haircare",
            "Makeup",
            "Fragrance",
            "Personal Care",
        ],
        "brands": [
            "Glow",
            "PureCare",
            "BeautyPlus",
            "NatureTouch",
            "Aura",
        ],
    },
    "Gaming": {
        "subcategories": [
            "Console",
            "Games",
            "Controller",
            "Gaming Audio",
            "Accessories",
        ],
        "brands": [
            "GameZone",
            "PlayMax",
            "XCore",
            "PixelPro",
            "GameTech",
        ],
    },
    "Grocery": {
        "subcategories": [
            "Snacks",
            "Beverages",
            "Breakfast",
            "Packaged Food",
            "Organic",
        ],
        "brands": [
            "FreshChoice",
            "DailyBasket",
            "NatureBest",
            "GoodFood",
            "Harvest",
        ],
    },
}


CITIES = [
    "Bengaluru",
    "Mumbai",
    "Delhi",
    "Hyderabad",
    "Chennai",
    "Pune",
    "Kolkata",
    "Ahmedabad",
    "Jaipur",
    "Bhubaneswar",
]

GENDERS = [
    "Male",
    "Female",
    "Other",
]

MEMBERSHIP_TIERS = [
    "Bronze",
    "Silver",
    "Gold",
    "Platinum",
]

CUSTOMER_SEGMENTS = [
    "Frequent_Buyer",
    "Occasional_Buyer",
    "Value_Seeker",
    "Premium_Buyer",
    "Window_Shopper",
]

PRICE_PREFERENCES = [
    "Budget",
    "Mid",
    "Premium",
]

INTERACTION_TYPES = [
    "view",
    "cart",
    "wishlist",
    "purchase",
    "rating",
]

BASE_INTERACTION_WEIGHTS = {
    "view": 0.10,
    "wishlist": 0.35,
    "cart": 0.60,
    "purchase": 1.00,
    "rating": 0.80,
}


# ============================================================
# Review vocabulary
# ============================================================

POSITIVE_OPENINGS = [
    "I am quite happy with this purchase.",
    "Overall, this turned out to be a good purchase.",
    "My experience with this product has been positive.",
    "I have been using this product regularly and like it.",
    "The product has worked well for me so far.",
    "I was pleasantly surprised by this product.",
]

NEUTRAL_OPENINGS = [
    "My experience with this product has been fairly average.",
    "The product is reasonable for its price.",
    "This product works, although it has some limitations.",
    "My experience has been mixed so far.",
    "The product performs its basic function adequately.",
    "There are both good and average aspects to this product.",
]

NEGATIVE_OPENINGS = [
    "My experience with this product has been disappointing.",
    "I expected more from this purchase.",
    "The product has not worked as well as I expected.",
    "I have had several issues with this product.",
    "Unfortunately, this purchase did not work out for me.",
    "There are some noticeable problems with this product.",
]


POSITIVE_PHRASES = [
    "The quality feels solid.",
    "The performance has been reliable.",
    "The design is practical and appealing.",
    "It has been convenient to use.",
    "The product performs well during regular use.",
    "The overall value is good.",
    "The build quality is better than expected.",
    "The main features work smoothly.",
]

NEUTRAL_PHRASES = [
    "The quality is acceptable.",
    "The performance is reasonable.",
    "The design is fairly practical.",
    "It works as expected for normal use.",
    "The product is reasonably easy to use.",
    "The value is acceptable for the price.",
    "The main features work adequately.",
    "Some aspects are better than others.",
]

NEGATIVE_PHRASES = [
    "The quality could be better.",
    "The performance is inconsistent.",
    "The design has some practical issues.",
    "It is not always convenient to use.",
    "The product has not been very reliable.",
    "The value does not feel convincing.",
    "The build quality could be improved.",
    "Some of the main features have problems.",
]


POSITIVE_ENDINGS = [
    "I would consider buying it again.",
    "I would recommend it for normal use.",
    "It has been a worthwhile purchase.",
    "I am satisfied with the overall experience.",
]

NEUTRAL_ENDINGS = [
    "It does the basic job without standing out.",
    "It is suitable if expectations are reasonable.",
    "There is still some room for improvement.",
    "I would describe the experience as average.",
]

NEGATIVE_ENDINGS = [
    "I would probably choose another product next time.",
    "I would not recommend it without reservations.",
    "I expected better overall performance.",
    "There is considerable room for improvement.",
]


MIXED_PHRASES = [
    "The design is good, although the performance could be better.",
    "The product is useful, but a few details are frustrating.",
    "The main feature works well, while some smaller features need improvement.",
    "The quality is reasonable, although the value could be better.",
    "It performs well in some situations but has noticeable limitations.",
]


# ============================================================
# Utility functions
# ============================================================

def weighted_choice(values, weights):
    return random.choices(
        values,
        weights=weights,
        k=1,
    )[0]


def random_date(start_date, end_date):
    days = (end_date - start_date).days

    return start_date + timedelta(
        days=random.randint(0, days)
    )


def price_range_for_preference(preference):
    if preference == "Budget":
        return random.uniform(100, 1500)

    if preference == "Mid":
        return random.uniform(1000, 8000)

    return random.uniform(5000, 50000)


def price_match_score(price, preference):
    if preference == "Budget":
        if price <= 2000:
            return 1.0

        if price <= 5000:
            return 0.5

        return 0.2

    if preference == "Mid":
        if 1000 <= price <= 10000:
            return 1.0

        if price < 1000 or price <= 20000:
            return 0.5

        return 0.2

    if price >= 5000:
        return 1.0

    if price >= 2500:
        return 0.5

    return 0.2


# ============================================================
# Customer-product relevance
# ============================================================

def category_relevance(product_category, customer):
    if product_category == customer["preferred_category"]:
        return 1.0

    if product_category == customer["secondary_category"]:
        return 0.65

    return 0.25


def calculate_product_relevance(product, customer):
    category_score = category_relevance(
        product["category"],
        customer,
    )

    price_score = price_match_score(
        product["price"],
        customer["price_preference"],
    )

    popularity_score = product["popularity_score"]

    relevance = (
        0.60 * category_score
        + 0.25 * price_score
        + 0.15 * popularity_score
    )

    # Small noise prevents perfectly deterministic ranking.
    relevance += random.uniform(
        -0.08,
        0.08,
    )

    return max(
        0.05,
        min(
            1.0,
            relevance,
        ),
    )


# ============================================================
# Review sentiment
# ============================================================

def sentiment_from_rating(rating):
    random_value = random.random()

    if rating >= 4:
        if random_value < 0.72:
            return "positive"

        if random_value < 0.90:
            return "neutral"

        return "negative"

    if rating <= 2:
        if random_value < 0.72:
            return "negative"

        if random_value < 0.90:
            return "neutral"

        return "positive"

    # Rating == 3
    if random_value < 0.60:
        return "neutral"

    if random_value < 0.80:
        return "positive"

    return "negative"


def build_review_text(product, experience_score):
    feature = product["subcategory"].lower()
    brand = product["brand"]

    openings = [
        f"I have been using this {feature} product for a while.",
        f"I bought this {feature} product recently.",
        f"My experience with this {feature} product has been mixed.",
        f"I have used this {feature} item several times.",
        f"This {brand} product has been part of my regular use.",
        f"I tried this product mainly for its {feature} features.",
        f"I have had enough time to get a feel for this product.",
        f"This product seemed interesting, so I decided to try it.",
        f"I picked up this product after comparing a few options.",
        f"I have been testing this product in my daily routine.",
        f"I decided to give this product a try.",
        f"This was one of the products I considered in this category.",
    ]

    common_aspects = [
        "The design is practical.",
        "The product is fairly easy to use.",
        "The quality is reasonable.",
        "The main features work as expected.",
        "The product looks well designed.",
        "The overall experience is acceptable.",
        "The performance is noticeable during regular use.",
        "The product has some useful features.",
        "The controls are straightforward.",
        "The size works well for normal use.",
        "The materials feel fairly standard.",
        "The setup process was manageable.",
        "The product fits its intended purpose.",
        "The overall design is fairly simple.",
    ]

    positive_aspects = [
        "The performance has been reliable.",
        "The quality is better than I expected.",
        "It has been convenient for everyday use.",
        "The main features work smoothly.",
        "The overall value is good.",
        "The build quality feels solid.",
    ]

    negative_aspects = [
        "The performance has been inconsistent.",
        "The quality could be improved.",
        "Some features are less useful than expected.",
        "The product could offer better value.",
        "There are a few noticeable issues.",
        "The experience has not been completely reliable.",
    ]

    mild_positive = [
        "The product works reasonably well.",
        "Some parts of the experience are quite good.",
        "The main functionality is useful.",
        "The product has performed fairly well.",
    ]

    mild_negative = [
        "Some parts could work better.",
        "There are a few things I would change.",
        "The product has some limitations.",
        "A few details could be improved.",
    ]

    opening = random.choice(openings)

    # The review generator receives experience_score,
    # not the final sentiment label.

    if experience_score >= 0.75:
        aspect = random.choice(
            positive_aspects
        )

    elif experience_score >= 0.55:
        aspect = random.choice(
            mild_positive
        )

    elif experience_score >= 0.45:
        aspect = random.choice(
            common_aspects
        )

    elif experience_score >= 0.25:
        aspect = random.choice(
            mild_negative
        )

    else:
        aspect = random.choice(
            negative_aspects
        )

    # Add a second aspect sometimes.
    if random.random() < 0.65:

        second_aspect = random.choice(
            common_aspects
            + mild_positive
            + mild_negative
        )

    else:
        second_aspect = ""

    # Occasionally create a mixed opinion.
    if random.random() < 0.35:

        if experience_score >= 0.50:
            contrast = random.choice(
                mild_negative
            )
        else:
            contrast = random.choice(
                mild_positive
            )

    else:
        contrast = ""

    parts = [
        opening,
        aspect,
    ]

    if second_aspect:
        parts.append(
            second_aspect
        )

    if contrast:
        parts.append(
            "However, " + contrast.lower()
        )

    return " ".join(parts)
# ============================================================
# Customer generation
# ============================================================

def generate_customers():
    categories = list(
        CATEGORY_CONFIG.keys()
    )

    customers = []

    for i in range(
        1,
        NUM_CUSTOMERS + 1,
    ):
        preferred_category = random.choice(
            categories
        )

        remaining_categories = [
            category
            for category in categories
            if category != preferred_category
        ]

        secondary_category = random.choice(
            remaining_categories
        )

        age = int(
            np.clip(
                np.random.normal(
                    32,
                    9,
                ),
                18,
                65,
            )
        )

        customers.append(
            {
                "customer_id": f"CUST{i:05d}",
                "age": age,
                "gender": random.choice(
                    GENDERS
                ),
                "city": random.choice(
                    CITIES
                ),
                "membership_tier": weighted_choice(
                    MEMBERSHIP_TIERS,
                    [45, 30, 18, 7],
                ),
                "preferred_category": preferred_category,
                "secondary_category": secondary_category,
                "price_preference": weighted_choice(
                    PRICE_PREFERENCES,
                    [35, 45, 20],
                ),
                "customer_segment": weighted_choice(
                    CUSTOMER_SEGMENTS,
                    [25, 35, 15, 10, 15],
                ),
            }
        )

    return pd.DataFrame(customers)


# ============================================================
# Product generation
# ============================================================

def generate_products():
    categories = list(
        CATEGORY_CONFIG.keys()
    )

    products = []

    for i in range(
        1,
        NUM_PRODUCTS + 1,
    ):
        category = random.choice(
            categories
        )

        config = CATEGORY_CONFIG[
            category
        ]

        subcategory = random.choice(
            config["subcategories"]
        )

        brand = random.choice(
            config["brands"]
        )

        price = round(
            price_range_for_preference(
                weighted_choice(
                    PRICE_PREFERENCES,
                    [35, 45, 20],
                )
            ),
            2,
        )

        average_rating = round(
            float(
                np.clip(
                    np.random.normal(
                        4.0,
                        0.55,
                    ),
                    2.0,
                    5.0,
                )
            ),
            1,
        )

        review_count = int(
            np.random.lognormal(
                mean=4.0,
                sigma=0.8,
            )
        )

        popularity_score = round(
            float(
                np.clip(
                    np.random.beta(
                        4,
                        2,
                    ),
                    0,
                    1,
                )
            ),
            3,
        )

        stock_status = weighted_choice(
            [
                "In Stock",
                "Low Stock",
                "Out of Stock",
            ],
            [80, 15, 5],
        )

        products.append(
            {
                "product_id": f"PROD{i:05d}",
                "product_name": (
                    f"{brand} "
                    f"{subcategory} "
                    f"Product {i:03d}"
                ),
                "category": category,
                "subcategory": subcategory,
                "brand": brand,
                "price": price,
                "average_rating": average_rating,
                "review_count": review_count,
                "popularity_score": popularity_score,
                "stock_status": stock_status,
            }
        )

    return pd.DataFrame(products)


# ============================================================
# Interaction volume
# ============================================================

def generate_interaction_counts(customers):
    counts = {}

    for customer in customers:
        random_value = random.random()

        if random_value < 0.05:
            # Cold-start customer.
            count = 0

        elif random_value < 0.15:
            # Limited-history customer.
            count = random.randint(
                1,
                3,
            )

        else:
            # Normal customer.
            count = max(
                8,
                int(
                    np.random.normal(
                        22,
                        6,
                    )
                ),
            )

        counts[
            customer["customer_id"]
        ] = count

    return counts


# ============================================================
# Interaction type probabilities
# ============================================================

def get_interaction_type_weights(
    relevance,
):
    # Higher relevance should increase the probability
    # of meaningful interactions such as cart/purchase/rating.

    view_weight = 55 - (20 * relevance)
    cart_weight = 10 + (10 * relevance)
    wishlist_weight = 8 + (4 * relevance)
    purchase_weight = 15 + (20 * relevance)
    rating_weight = 5 + (6 * relevance)

    return [
        view_weight,
        cart_weight,
        wishlist_weight,
        purchase_weight,
        rating_weight,
    ]


# ============================================================
# Interaction generation
# ============================================================

def generate_interactions(
    customers_df,
    products_df,
):
    customers = customers_df.to_dict(
        "records"
    )

    products = products_df.to_dict(
        "records"
    )

    interaction_counts = (
        generate_interaction_counts(
            customers
        )
    )

    total_requested = sum(
        interaction_counts.values()
    )

    # Scale the counts to approximately the requested
    # total number of interactions.
    if total_requested > 0:
        scale = (
            NUM_INTERACTIONS
            / total_requested
        )
    else:
        scale = 1

    scaled_counts = {}

    for customer_id, count in interaction_counts.items():
        if count == 0:
            scaled_counts[customer_id] = 0
        else:
            scaled_counts[customer_id] = max(
                1,
                round(count * scale),
            )

    # Ensure exactly NUM_INTERACTIONS interactions.
    current_total = sum(
        scaled_counts.values()
    )

    difference = (
        NUM_INTERACTIONS
        - current_total
    )

    active_customers = [
        customer_id
        for customer_id, count
        in scaled_counts.items()
        if count > 0
    ]

    if difference > 0:
        for _ in range(difference):
            customer_id = random.choice(
                active_customers
            )

            scaled_counts[customer_id] += 1

    else:
        removable_customers = [
            customer_id
            for customer_id in active_customers
            if scaled_counts[customer_id] > 1
        ]

        for _ in range(abs(difference)):
            customer_id = random.choice(
                removable_customers
            )

            scaled_counts[customer_id] -= 1

            if scaled_counts[customer_id] <= 1:
                removable_customers.remove(
                    customer_id
                )

    interactions = []

    start_date = datetime(
        2025,
        1,
        1,
    )

    end_date = datetime(
        2026,
        8,
        1,
    )

    interaction_id = 1

    product_lookup = {
        product["product_id"]: product
        for product in products
    }

    for customer in customers:
        customer_id = customer[
            "customer_id"
        ]

        number_of_interactions = (
            scaled_counts.get(
                customer_id,
                0,
            )
        )

        if number_of_interactions == 0:
            continue

        for _ in range(
            number_of_interactions
        ):
            # Sample a small candidate set.
            candidate_products = random.sample(
                products,
                min(
                    25,
                    len(products),
                ),
            )

            scored_products = []

            for product in candidate_products:
                relevance = (
                    calculate_product_relevance(
                        product,
                        customer,
                    )
                )

                # Add randomness so the highest relevance
                # product is not selected every time.
                selection_score = (
                    relevance
                    + random.uniform(
                        0,
                        0.35,
                    )
                )

                scored_products.append(
                    (
                        product,
                        relevance,
                        selection_score,
                    )
                )

            # Sort by selection score.
            scored_products.sort(
                key=lambda item: item[2],
                reverse=True,
            )

            # Pick from the top candidates rather than
            # always taking the single best product.
            top_candidates = (
                scored_products[:8]
            )

            selected = random.choice(
                top_candidates
            )

            product = selected[0]
            relevance = selected[1]

            interaction_type = weighted_choice(
                INTERACTION_TYPES,
                get_interaction_type_weights(
                    relevance
                ),
            )

            # Interaction score now depends on relevance.
            base_score = BASE_INTERACTION_WEIGHTS[
                interaction_type
            ]

            interaction_score = (
                base_score
                * (
                    0.65
                    + 0.55 * relevance
                )
            )

            interaction_score += random.uniform(
                -0.05,
                0.05,
            )

            interaction_score = round(
                max(
                    0.05,
                    min(
                        1.0,
                        interaction_score,
                    ),
                ),
                3,
            )

            rating = np.nan

            if interaction_type == "rating":
                rating_mean = (
                    2.5
                    + (
                        2.0
                        * relevance
                    )
                )

                rating = int(
                    np.clip(
                        round(
                            np.random.normal(
                                rating_mean,
                                0.75,
                            )
                        ),
                        1,
                        5,
                    )
                )

            quantity = 1

            if interaction_type == "purchase":
                quantity = weighted_choice(
                    [1, 2, 3],
                    [80, 17, 3],
                )

            interactions.append(
                {
                    "interaction_id": (
                        f"INT{interaction_id:07d}"
                    ),
                    "customer_id": customer_id,
                    "product_id": product[
                        "product_id"
                    ],
                    "interaction_type": (
                        interaction_type
                    ),
                    "interaction_score": (
                        interaction_score
                    ),
                    "rating": rating,
                    "quantity": quantity,
                    "timestamp": random_date(
                        start_date,
                        end_date,
                    ),
                }
            )

            interaction_id += 1

    return pd.DataFrame(
        interactions
    )


# ============================================================
# Review generation
# ============================================================

def generate_reviews(
    customers_df,
    products_df,
    interactions_df,
):
    customers = (
        customers_df
        .set_index("customer_id")
        .to_dict("index")
    )

    products = (
        products_df
        .set_index("product_id")
        .to_dict("index")
    )

    purchase_or_rating = interactions_df[
        interactions_df[
            "interaction_type"
        ].isin(
            [
                "purchase",
                "rating",
            ]
        )
    ]

    if len(purchase_or_rating) < NUM_REVIEWS:
        review_source = interactions_df
    else:
        review_source = purchase_or_rating

    review_source = review_source.sample(
        n=NUM_REVIEWS,
        random_state=SEED,
        replace=(
            len(review_source)
            < NUM_REVIEWS
        ),
    )

    reviews = []

    for i, row in enumerate(
        review_source.itertuples(
            index=False
        ),
        start=1,
    ):
        customer = customers[
            row.customer_id
        ]

        product = products[
            row.product_id
        ]

        # Generate ratings with a slightly wider
        # distribution than the previous version.
        rating = int(
            np.clip(
                round(
                    np.random.normal(
                        product[
                            "average_rating"
                        ],
                        0.95,
                    )
                ),
                1,
                5,
            )
        )

        # --------------------------------------------------------
        # Generate a latent customer experience.
        # This is intentionally different from directly
        # generating text from the sentiment label.
        # --------------------------------------------------------

        rating_signal = (
            rating - 1
        ) / 4.0

        experience_score = (
            0.55 * rating_signal
            + 0.45 * random.random()
        )

        experience_score = max(
            0.0,
            min(
                1.0,
                experience_score,
            ),
        )

        # Small amount of natural variation.
        experience_score += random.uniform(
            -0.10,
            0.10,
        )

        experience_score = max(
            0.0,
            min(
                1.0,
                experience_score,
            ),
        )

        # Generate review text from the underlying experience,
        # NOT from the final sentiment label.

        review_text = build_review_text(
            product,
            experience_score,
        )

        # Sentiment is derived separately from the
        # underlying experience.

        if experience_score >= 0.58:

            sentiment = "positive"

        elif experience_score <= 0.35:

            sentiment = "negative"

        else:

            sentiment = "neutral"

        title_map = {
            "positive": [
                "Good product",
                "Satisfied overall",
                "Worth considering",
                "Good experience",
                "Works well",
            ],
            "neutral": [
                "Average experience",
                "Works as expected",
                "Decent product",
                "Fair experience",
                "Acceptable overall",
            ],
            "negative": [
                "Could be better",
                "Disappointing experience",
                "Needs improvement",
                "Not fully satisfied",
                "Several issues",
            ],
        }

        reviews.append(
            {
                "review_id": (
                    f"REV{i:06d}"
                ),
                "customer_id": row.customer_id,
                "product_id": row.product_id,
                "rating": rating,
                "review_title": random.choice(
                    title_map[sentiment]
                ),
                "review_text": review_text,
                "verified_purchase": (
                    random.random()
                    < 0.78
                ),
                "review_date": random_date(
                    datetime(
                        2025,
                        1,
                        1,
                    ),
                    datetime(
                        2026,
                        8,
                        15,
                    ),
                ),
                "sentiment": sentiment,
            }
        )

    return pd.DataFrame(
        reviews
    )


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Generating customers...")
    customers_df = generate_customers()

    print("Generating products...")
    products_df = generate_products()

    print("Generating interactions...")
    interactions_df = generate_interactions(
        customers_df,
        products_df,
    )

    print("Generating reviews...")
    reviews_df = generate_reviews(
        customers_df,
        products_df,
        interactions_df,
    )

    customers_df.to_csv(
        OUTPUT_DIR / "customers.csv",
        index=False,
    )

    products_df.to_csv(
        OUTPUT_DIR / "products.csv",
        index=False,
    )

    interactions_df.to_csv(
        OUTPUT_DIR / "interactions.csv",
        index=False,
    )

    reviews_df.to_csv(
        OUTPUT_DIR / "reviews.csv",
        index=False,
    )

    print()
    print(
        "Dataset generation completed."
    )
    print()

    print(
        f"Customers:    {len(customers_df):,}"
    )

    print(
        f"Products:     {len(products_df):,}"
    )

    print(
        f"Interactions: {len(interactions_df):,}"
    )

    print(
        f"Reviews:      {len(reviews_df):,}"
    )

    print()
    print(
        f"Files saved to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()