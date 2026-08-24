from recommender import ProductRecommender


def main():

    recommender = (
        ProductRecommender()
    )

    # --------------------------------------------------------
    # Existing customer
    # --------------------------------------------------------

    customer_id = "CUST00001"

    print()
    print("=" * 70)
    print("EXISTING CUSTOMER RECOMMENDATIONS")
    print("=" * 70)

    recommendations = (
        recommender.recommend(
            customer_id,
            top_n=10,
        )
    )

    print(
        recommendations.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Find a cold-start customer
    # --------------------------------------------------------

    interaction_counts = (
        recommender.interactions
        .groupby(
            "customer_id"
        )
        .size()
    )

    cold_start_customers = [
        customer
        for customer in recommender.customers[
            "customer_id"
        ]
        if customer
        not in interaction_counts.index
    ]

    if cold_start_customers:

        cold_customer = (
            cold_start_customers[0]
        )

        print()
        print("=" * 70)
        print(
            "COLD-START CUSTOMER RECOMMENDATIONS"
        )
        print("=" * 70)

        recommendations = (
            recommender.recommend(
                cold_customer,
                top_n=10,
            )
        )

        print(
            recommendations.to_string(
                index=False
            )
        )

    else:

        print()
        print(
            "No cold-start customers "
            "exist in the current dataset."
        )


if __name__ == "__main__":
    main()