from service import get_recommendations


def main():
    customer_id = "CUST00001"

    recommendations = get_recommendations(
        customer_id,
        top_n=5,
    )

    print("=" * 70)
    print("RECOMMENDATION SERVICE TEST")
    print("=" * 70)
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()