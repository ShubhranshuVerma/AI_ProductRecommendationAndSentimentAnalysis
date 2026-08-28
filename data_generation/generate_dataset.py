import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


  # Configuration
  
SEED = 42

NUM_CUSTOMERS = 1000
NUM_PRODUCTS = 250
NUM_INTERACTIONS = 20000
NUM_REVIEWS = 5000

OUTPUT_DIR = Path("data/raw")

random.seed(SEED)
np.random.seed(SEED)


  # Product configuration
  
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


  # Review vocabulary
  
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


  # Utility functions
  
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


  # Customer-product relevance
  
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


  # Review sentiment
  
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
    Generate diverse, natural product reviews from a latent
    customer experience score.

    The generator intentionally includes:
    - direct sentiment
    - implicit sentiment
    - negation
    - contrast
    - expectation mismatch
    - recommendation language
    - temporal changes
    - mixed opinions
    - varied sentence structures

    This is designed to improve generalization to natural
    user-written reviews rather than only synthetic templates.
    """

    category = product["subcategory"].lower()
    brand = product["brand"]

    # --------------------------------------------------------
    # Neutral contextual openings
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
    # Positive observations
    # --------------------------------------------------------

    positive = [
        "It works exactly as I hoped it would.",
        "The overall experience has been excellent.",
        "I am very happy with how it performs.",
        "The product has exceeded my expectations.",
        "Everything has worked smoothly so far.",
        "The quality feels better than I expected.",
        "It has made my daily routine easier.",
        "I have had a very positive experience with it.",
        "The performance has been consistently reliable.",
        "I would happily recommend this product.",
        "I am genuinely satisfied with the purchase.",
        "It has turned out to be a great choice.",
        "The product feels well made and dependable.",
        "I have been impressed by how well it performs.",
        "It has handled everything I have asked it to do.",
        "I would definitely consider buying it again.",
    ]

    # --------------------------------------------------------
    # Negative observations
    # --------------------------------------------------------

    negative = [
        "The product has been disappointing overall.",
        "The performance has not lived up to my expectations.",
        "I have experienced several problems during normal use.",
        "The product has become unreliable over time.",
        "It has been more frustrating than useful.",
        "The quality is worse than I expected.",
        "I have had problems that should not occur during normal use.",
        "The product does not perform consistently.",
        "I regret choosing this product.",
        "I would not recommend it to other buyers.",
        "The experience has been noticeably worse than expected.",
        "Several important features have not worked properly.",
        "I expected much better performance for the price.",
        "The product has caused more problems than it solved.",
        "I would choose a different option next time.",
        "I am not satisfied with the purchase.",
    ]

    # --------------------------------------------------------
    # Negation / linguistic nuance patterns
    # --------------------------------------------------------

    positive_nuance = [
        "The product is not bad at all.",
        "I cannot say anything negative about it.",
        "I did not expect to like it this much.",
        "I was not expecting such good performance.",
        "It is much better than I expected.",
        "I am extremely satisfied with the purchase.",
        "I am genuinely impressed with the overall experience.",
        "I absolutely love how well this product works.",
        "I could not be happier with the purchase.",
        "There is very little I would change about it.",
    ]

    negative_nuance = [
        "The product is not good enough for my needs.",
        "I cannot recommend this product.",
        "I did not expect the quality to be this poor.",
        "It is much worse than I expected.",
        "I am extremely disappointed with the purchase.",
        "I am genuinely unhappy with the overall experience.",
        "I regret buying this product.",
        "I would not recommend this product to anyone.",
        "I could not justify buying this again.",
        "There is very little I like about the product.",
    ]

    neutral_nuance = [
        "The product is not bad, but it is not impressive either.",
        "The product is not particularly good or bad.",
        "I cannot say that it is either excellent or terrible.",
        "It is neither impressive nor disappointing.",
        "I expected it to be average, and that is exactly what it is.",
        "It is acceptable, although I would not call it excellent.",
        "The product is okay, but nothing special.",
        "It is not outstanding, but it gets the job done.",
        "There are things I like and things I would change.",
        "The experience is difficult to describe as either very good or very bad.",
    ]

    # --------------------------------------------------------
    # Neutral observations
    # --------------------------------------------------------

    neutral = [
        "The product works adequately for normal use.",
        "It does what it is supposed to do, but nothing more.",
        "The overall experience has been fairly ordinary.",
        "It is acceptable, although there is room for improvement.",
        "Some parts work well while others are average.",
        "The product meets the basic requirements.",
        "I have not found anything particularly impressive about it.",
        "It performs reasonably well for everyday tasks.",
        "There are both useful features and noticeable limitations.",
        "The experience is neither especially good nor especially bad.",
        "It is functional without being exceptional.",
        "I would describe the overall quality as average.",
        "The product is fine if expectations remain reasonable.",
        "It has been usable so far, but I am not particularly impressed.",
    ]

    # --------------------------------------------------------
    # Natural linguistic patterns
    # --------------------------------------------------------

    positive_patterns = [
        "The battery is excellent and the overall performance is impressive.",
        "I expected this to be average, but it turned out to be much better.",
        "I was initially unsure about it, but I have become very satisfied with it.",
        "There is very little I would change about the product.",
        "Even after repeated use, it has continued to perform reliably.",
        "I was pleasantly surprised by how good the product is.",
        "I cannot complain about the performance so far.",
    ]

    negative_patterns = [
        "The product looked promising at first, but the problems became obvious later.",
        "I expected much better from a product in this price range.",
        "It is not something I would recommend to a friend.",
        "The product is not good enough for my needs.",
        "At first everything seemed fine, but the performance deteriorated over time.",
        "The problems are difficult to overlook.",
        "I wanted to like this product, but the issues have changed my opinion.",
    ]

    neutral_patterns = [
        "It is not bad, but it is not particularly impressive either.",
        "The product works, although I would not describe the experience as exceptional.",
        "There are some positives, but nothing that makes it stand out.",
        "It is neither a great product nor a terrible one.",
        "The main functions work, while some smaller details could be improved.",
        "It does the basic job without offering anything remarkable.",
    ]

    mixed_patterns = [
        "The battery is excellent, but the camera is disappointing.",
        "The display is excellent, although the performance could be better.",
        "The product is well designed, but its reliability is disappointing.",
        "The performance is impressive, but the battery life is poor.",
        "The build quality is excellent, although some features are unreliable.",
        "I like several things about this product, but there are also some significant drawbacks.",
        "The product has some excellent features, but the weaknesses are difficult to ignore.",
        "Some aspects are genuinely impressive, while others are disappointing.",
    ]

    # --------------------------------------------------------
    # Contrast / mixed sentiment
    # --------------------------------------------------------

    positive_contrast = [
        "The battery is excellent, although the design could be improved.",
        "The product has a few minor flaws, but overall I am very satisfied.",
        "The setup took some effort, but the product has performed very well since then.",
        "There are small limitations, but they do not affect my overall satisfaction.",
    ]

    negative_contrast = [
        "The design looks good, but the actual performance is disappointing.",
        "The product has some useful features, but the reliability is poor.",
        "The battery is decent, but the device becomes frustrating to use under heavy workloads.",
        "There are things I like about it, but the problems outweigh the benefits.",
    ]

    # --------------------------------------------------------
    # Temporal patterns
    # --------------------------------------------------------

    positive_temporal = [
        "It has remained reliable even after several weeks of use.",
        "The performance has stayed consistent over time.",
        "After using it regularly, I am even more satisfied with it.",
    ]

    negative_temporal = [
        "It started developing problems after only a few weeks.",
        "The performance became worse after repeated use.",
        "It worked well initially, but the problems appeared over time.",
    ]

    # --------------------------------------------------------
    # Recommendation patterns
    # --------------------------------------------------------

    positive_recommendations = [
        "I would recommend it without hesitation.",
        "I would happily buy this product again.",
        "I would choose this over several alternatives.",
    ]

    negative_recommendations = [
        "I would not recommend this product.",
        "I would look for another option next time.",
        "I would not buy this product again.",
    ]

    # --------------------------------------------------------
    # Determine semantic region
    # --------------------------------------------------------

    if experience_score >= 0.65:
        primary_pool = positive
        pattern_pool = positive_patterns
        contrast_pool = positive_contrast
        temporal_pool = positive_temporal
        recommendation_pool = positive_recommendations

    elif experience_score <= 0.35:
        primary_pool = negative
        pattern_pool = negative_patterns
        contrast_pool = negative_contrast
        temporal_pool = negative_temporal
        recommendation_pool = negative_recommendations

    else:
        primary_pool = neutral
        pattern_pool = neutral_patterns
        contrast_pool = [
            "It has some useful features, although there is still room for improvement.",
            "The main functions work well enough, but some smaller details could be refined.",
            "It is reasonably capable, although I would not describe it as exceptional.",
            "The product has advantages and limitations, and neither stands out strongly.",
            "The basic experience is fine, but there are no particularly impressive features.",
        ]
        temporal_pool = (
            positive_temporal
            + negative_temporal
        )
        recommendation_pool = [
            "I would consider a few alternatives before buying it again.",
            "It is a reasonable option for basic requirements.",
            "I would keep my expectations realistic if choosing this product.",
        ]

    # --------------------------------------------------------
    # Choose review structure
    # --------------------------------------------------------

    structure = random.random()

    if structure < 0.20:
        # Short natural review
        sentences = [
            random.choice(primary_pool)
        ]

    elif structure < 0.40:
        # Context + opinion
        sentences = [
            random.choice(openings),
            random.choice(primary_pool),
        ]

    elif structure < 0.60:
        # Opinion + supporting observation
        sentences = [
            random.choice(primary_pool),
            random.choice(pattern_pool),
        ]

    elif structure < 0.80:
        # Context + contrast
        sentences = [
            random.choice(openings),
            random.choice(contrast_pool),
        ]

    else:
        # Longer natural review
        sentences = [
            random.choice(openings),
            random.choice(primary_pool),
            random.choice(temporal_pool),
        ]

    # --------------------------------------------------------
    # Add recommendation language sometimes
    # --------------------------------------------------------

    if random.random() < 0.30:
        sentences.append(
            random.choice(recommendation_pool)
        )


    if random.random() < 0.12:
        sentences.append(
            random.choice(mixed_patterns)
        )

    # --------------------------------------------------------
    # Add controlled mixed sentiment
    # --------------------------------------------------------

    if random.random() < 0.15:

        if experience_score >= 0.65:
            sentences.append(
                random.choice([
                    "There are still a few minor details that could be improved.",
                    "It is not completely perfect, but the positives clearly outweigh the negatives.",
                    "A few small limitations do not change my overall positive impression.",
                ])
            )

        elif experience_score <= 0.35:
            sentences.append(
                random.choice([
                    "One or two aspects are useful, but the main problems remain.",
                    "There are a few good qualities, but they do not outweigh the problems.",
                    "Some parts are acceptable, although the overall experience is disappointing.",
                ])
            )

        else:
            sentences.append(
                random.choice([
                    "Some parts are better than others.",
                    "I can see both advantages and disadvantages.",
                    "Overall, the experience is fairly balanced.",
                ])
            )

    return " ".join(sentences)
  # Customer generation
  
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


  # Product generation
  
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


  # Interaction volume
  
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


  # Interaction type probabilities
  
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


  # Interaction generation
  
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


  # Review generation
  
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

        # --------------------------------------------------------
        # Generate a balanced sentiment label independently
        # from the product rating.
        #
        # Rating remains a realistic product attribute, while
        # sentiment is deliberately balanced for ML training.
        # --------------------------------------------------------

        sentiment = random.choice(
            [
                "positive",
                "neutral",
                "negative",
            ]
        )

        # Map the sentiment to an experience region so that
        # review text generation and sentiment labels remain
        # semantically aligned.
        experience_ranges = {
            "positive": (0.70, 1.00),
            "neutral": (0.40, 0.60),
            "negative": (0.00, 0.30),
        }

        low, high = experience_ranges[sentiment]

        experience_score = random.uniform(
            low,
            high,
        )

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
        # Generate review text from the sentiment-aligned
        # experience score.

        for _ in range(20):
            review_text = build_review_text(
                product,
                experience_score,
            )

            if review_text not in used_review_texts:
                break
        else:
            review_text += f" Product reference {i}."

        used_review_texts.add(review_text)

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


  # Main
  
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