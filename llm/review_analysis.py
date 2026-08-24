import json

from llm.client import generate_response


# ============================================================
# Build review analysis prompt
# ============================================================

def build_analysis_prompt(
    product_name,
    reviews,
):
    review_text = "\n".join(
        [
            f"- Sentiment: {review['sentiment']} | "
            f"Review: {review['review_text']}"
            for review in reviews
        ]
    )

    prompt = f"""
You are an e-commerce customer feedback analyst.

Analyze customer reviews for the following product.

Product:
{product_name}

Customer reviews:
{review_text}

Provide a concise business-oriented analysis.

Return ONLY valid JSON using exactly this structure:

{{
    "summary": "Overall summary of customer feedback",
    "common_complaints": [
        "Complaint 1",
        "Complaint 2"
    ],
    "praised_features": [
        "Feature 1",
        "Feature 2"
    ],
    "business_insights": [
        "Insight 1",
        "Insight 2"
    ]
}}

Rules:

1. Base the analysis only on the supplied reviews.
2. Do not invent product characteristics.
3. Do not change the supplied sentiment labels.
4. Focus on recurring themes rather than individual opinions.
5. Keep each item concise.
6. If there is insufficient evidence for a category, return an empty list.
"""

    return prompt


# ============================================================
# Analyze reviews
# ============================================================

def analyze_reviews(
    product_name,
    reviews,
):
    if not reviews:
        raise ValueError(
            "At least one review is required."
        )

    prompt = build_analysis_prompt(
        product_name,
        reviews,
    )

    response = generate_response(
        prompt
    )

    return parse_analysis_response(
        response
    )


# ============================================================
# Parse LLM response
# ============================================================

def parse_analysis_response(
    response,
):
    response = response.strip()

    # Remove markdown JSON fences if the model
    # happens to return them.
    if response.startswith(
        "```json"
    ):
        response = response[
            7:
        ]

    if response.endswith(
        "```"
    ):
        response = response[
            :-3
        ]

    response = response.strip()

    try:
        result = json.loads(
            response
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            "LLM returned invalid JSON."
        ) from exc

    required_fields = [
        "summary",
        "common_complaints",
        "praised_features",
        "business_insights",
    ]

    for field in required_fields:

        if field not in result:
            raise ValueError(
                f"Missing LLM output field: "
                f"{field}"
            )

    return result