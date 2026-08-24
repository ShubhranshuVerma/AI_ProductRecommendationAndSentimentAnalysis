from review_analysis import analyze_reviews


def main():

    reviews = [
        {
            "sentiment": "positive",
            "review_text": (
                "The product quality is excellent "
                "and the design looks great."
            ),
        },
        {
            "sentiment": "positive",
            "review_text": (
                "Very good quality and easy to use."
            ),
        },
        {
            "sentiment": "negative",
            "review_text": (
                "The product stopped working "
                "after a few days."
            ),
        },
        {
            "sentiment": "negative",
            "review_text": (
                "The build quality could be much better."
            ),
        },
        {
            "sentiment": "neutral",
            "review_text": (
                "The product is okay but nothing special."
            ),
        },
    ]

    result = analyze_reviews(
        product_name="Sample Product",
        reviews=reviews,
    )

    print()
    print("=" * 70)
    print("LLM REVIEW ANALYSIS")
    print("=" * 70)

    print()
    print("Summary:")
    print(result["summary"])

    print()
    print("Common Complaints:")

    for item in result[
        "common_complaints"
    ]:
        print(f"- {item}")

    print()
    print("Praised Features:")

    for item in result[
        "praised_features"
    ]:
        print(f"- {item}")

    print()
    print("Business Insights:")

    for item in result[
        "business_insights"
    ]:
        print(f"- {item}")


if __name__ == "__main__":
    main()