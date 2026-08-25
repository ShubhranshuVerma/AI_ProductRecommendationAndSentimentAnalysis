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
    """
    Generate linguistically diverse review text from a latent
    customer experience score.

    The text is generated from multiple independent semantic
    components so the model cannot simply memorize a small set
    of sentiment phrases.
    """

    category = product["subcategory"].lower()
    brand = product["brand"]

    # --------------------------------------------------------
    # Neutral/contextual openings
    # --------------------------------------------------------

    openings = [
        f"I bought this {category} recently and have used it regularly.",
        f"I decided to try this {brand} product after comparing several options.",
        f"I have been using this {category} product for a few weeks.",
        f"After spending some time with this product, I have formed a fairly clear opinion.",
        f"I purchased this product mainly for its {category} features.",
        f"This product has been part of my regular routine for a while.",
        f"I have used this product in several everyday situations.",
        f"I gave this product a reasonable amount of use before writing this review.",
        f"I picked this product after looking at a few alternatives.",
        f"I have been testing this product under normal conditions.",
        f"After using the product for some time, I can comment on how it performs.",
        f"I bought this product to see how well it would fit my everyday needs.",
    ]

    # --------------------------------------------------------
    # Positive semantic expressions
    # --------------------------------------------------------

    positive_aspects = [
        "It has worked consistently without unexpected problems.",
        "The performance has remained dependable during regular use.",
        "I have found it easy and comfortable to use.",
        "The product handles its main tasks very well.",
        "The controls feel natural and easy to understand.",
        "The quality is better than I expected.",
        "It has made my routine noticeably easier.",
        "The product has remained reliable even after repeated use.",
        "I have not had any meaningful problems with it.",
        "The overall experience has been better than I anticipated.",
        "It performs reliably when I need it.",
        "The product feels thoughtfully designed.",
        "The setup was quick and straightforward.",
        "Everything has continued working as expected.",
        "I can depend on it for regular use.",
        "The product has been a pleasant surprise.",
        "Its main features have worked smoothly.",
        "The product feels well made.",
    ]

    # --------------------------------------------------------
    # Negative semantic expressions
    # --------------------------------------------------------

    negative_aspects = [
        "The product has become unreliable during normal use.",
        "I have encountered several problems that were difficult to ignore.",
        "The performance becomes inconsistent over longer periods.",
        "Some important functions have not worked properly.",
        "The materials do not feel particularly durable.",
        "The product has caused more frustration than I expected.",
        "I have experienced problems that should not occur during ordinary use.",
        "The quality does not match what I expected for the price.",
        "The product has failed to perform consistently.",
        "Several aspects of the experience could be substantially better.",
        "I have had trouble depending on it for regular use.",
        "The product started showing problems after repeated use.",
        "The actual performance falls short of its apparent potential.",
        "Some components already show signs of wear.",
        "The product has not been as dependable as I needed.",
        "I would hesitate to rely on this product for important tasks.",
        "The experience has been more disappointing than satisfying.",
        "There are several issues that make the product difficult to recommend.",
    ]

    # --------------------------------------------------------
    # Neutral semantic expressions
    # --------------------------------------------------------

    neutral_aspects = [
        "It handles the basic job adequately.",
        "The product works, although it does not stand out.",
        "The overall performance is acceptable for ordinary use.",
        "I have not found anything particularly impressive about it.",
        "There are useful aspects, but there are also some limitations.",
        "The product is functional without being exceptional.",
        "It performs roughly as I expected.",
        "The experience has been fairly ordinary overall.",
        "It gets the job done, but there is room for improvement.",
        "Some parts work well while others are fairly average.",
        "The product is suitable if expectations remain reasonable.",
        "I would describe the experience as acceptable rather than impressive.",
        "It meets the basic requirements without offering much beyond them.",
        "The product has both strengths and weaknesses.",
        "Nothing is seriously wrong, but I would not call it outstanding.",
        "The overall quality is reasonable for the price.",
        "It has been usable so far, although some details could be refined.",
        "The product is neither particularly impressive nor disappointing.",
    ]

    # --------------------------------------------------------
    # Positive supporting observations
    # --------------------------------------------------------

    positive_support = [
        "The setup was painless.",
        "I barely needed any time to get used to it.",
        "It has been dependable from the beginning.",
        "The day-to-day experience has been smooth.",
        "I have had no reason to complain about its reliability.",
        "It has handled regular use without difficulty.",
        "The overall design feels practical.",
        "It does what I need without unnecessary complications.",
    ]

    # --------------------------------------------------------
    # Negative supporting observations
    # --------------------------------------------------------

    negative_support = [
        "The problems became more noticeable with continued use.",
        "I expected something more dependable.",
        "The setup was more difficult than it needed to be.",
        "The issues are especially noticeable during longer sessions.",
        "I have had to work around several shortcomings.",
        "The product does not inspire much confidence.",
        "The problems affect the overall experience.",
        "I would prefer a more reliable alternative.",
    ]

    # --------------------------------------------------------
    # Neutral supporting observations
    # --------------------------------------------------------

    neutral_support = [
        "The differences are noticeable but not severe.",
        "The limitations are manageable for normal use.",
        "It depends somewhat on what you expect from the product.",
        "There are a few areas where refinement would help.",
        "The product is adequate for straightforward use.",
        "The strengths and weaknesses are fairly balanced.",
        "It is difficult to call the experience either excellent or poor.",
        "The result is reasonably predictable.",
    ]

    # --------------------------------------------------------
    # Positive endings
    # --------------------------------------------------------

    positive_endings = [
        "I would be comfortable recommending it.",
        "I would consider buying it again.",
        "Overall, I am satisfied with the purchase.",
        "It has turned out to be a worthwhile purchase.",
        "I am happy with how it has performed.",
        "I would choose this again over several alternatives.",
    ]

    # --------------------------------------------------------
    # Neutral endings
    # --------------------------------------------------------

    neutral_endings = [
        "I would describe the experience as average.",
        "It is fine as long as expectations are realistic.",
        "I would consider other options before making another purchase.",
        "Overall, it is acceptable but not exceptional.",
        "There is still some room for improvement.",
        "It is a reasonable choice for basic requirements.",
    ]

    # --------------------------------------------------------
    # Negative endings
    # --------------------------------------------------------

    negative_endings = [
        "I would probably choose another option next time.",
        "I would hesitate to recommend it.",
        "I expected better overall reliability.",
        "I do not think I would purchase it again.",
        "There is considerable room for improvement.",
        "I would look at alternatives before buying it again.",
    ]

    # --------------------------------------------------------
    # Determine semantic region
    # --------------------------------------------------------

    if experience_score >= 0.65:
        aspect_pool = positive_aspects
        support_pool = positive_support
        ending_pool = positive_endings

    elif experience_score <= 0.35:
        aspect_pool = negative_aspects
        support_pool = negative_support
        ending_pool = negative_endings

    else:
        aspect_pool = neutral_aspects
        support_pool = neutral_support
        ending_pool = neutral_endings

    # --------------------------------------------------------
    # Build review
    # --------------------------------------------------------

    sentences = [
        random.choice(openings),
        random.choice(aspect_pool),
    ]

    # Add supporting sentence most of the time.
    if random.random() < 0.70:
        sentences.append(
            random.choice(support_pool)
        )

    # Add a second semantic observation sometimes.
    if random.random() < 0.45:
        sentences.append(
            random.choice(aspect_pool)
        )

    # Add an ending sometimes.
    if random.random() < 0.55:
        sentences.append(
            random.choice(ending_pool)
        )

    # --------------------------------------------------------
    # Occasionally create controlled mixed reviews.
    # --------------------------------------------------------

    if random.random() < 0.15:

        if experience_score >= 0.65:
            sentences.append(
                "There are still a few minor details that could be improved."
            )

        elif experience_score <= 0.35:
            sentences.append(
                "One or two aspects of the product are still useful."
            )

        else:
            sentences.append(
                "Some parts are better than others."
            )

    return " ".join(sentences)
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

    used_review_texts = set()

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

        for _ in range(20):
            review_text = build_review_text(product, experience_score)
            if review_text not in used_review_texts:
                break
        else:
            review_text += f" Product reference {i}."
        used_review_texts.add(review_text)

        # Sentiment is derived separately from the
        # underlying experience.

        if experience_score >= 0.65:
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