import os

from dotenv import load_dotenv
from google import genai


# ============================================================
# Environment configuration
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash",
)


# ============================================================
# Create Gemini client
# ============================================================

def create_client():

    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not configured. "
            "Add it to your .env file."
        )

    return genai.Client(
        api_key=GEMINI_API_KEY
    )


# ============================================================
# Generate response
# ============================================================

def generate_response(prompt):

    client = create_client()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    if not response.text:
        raise RuntimeError(
            "LLM returned an empty response."
        )

    return response.text