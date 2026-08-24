from client import generate_response


def main():

    prompt = """
You are an assistant helping analyze customer product reviews.

Give a short response explaining why customer feedback
analysis is useful for an e-commerce business.

Keep the answer under 100 words.
"""

    response = generate_response(
        prompt
    )

    print()
    print("=" * 70)
    print("LLM TEST RESPONSE")
    print("=" * 70)
    print()
    print(response)


if __name__ == "__main__":
    main()