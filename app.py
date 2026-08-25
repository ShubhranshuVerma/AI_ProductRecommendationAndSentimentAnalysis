"""HOW TO RUN:
-----------------------------------------
1. Start the FastAPI Docker container:
docker run -d \
  -p 8001:8000 \
  --env-file .env \
  --name ai-product-recommendation \
  ai-product-recommendation

2. Start the Streamlit UI:
       streamlit run app.py"""
import os

import streamlit as st
import requests
import pandas as pd
from pathlib import Path

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL","http://127.0.0.1:8001",).rstrip("/")

# Project root = the folder that contains "data/", regardless of
# whether this file sits at project root or inside a ui/ folder.
PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

CUSTOMERS_PATH = PROJECT_ROOT / "data" / "processed" / "recommendation" / "customers.csv"
PRODUCTS_PATH = PROJECT_ROOT / "data" / "processed" / "recommendation" / "products.csv"
REVIEWS_PATH = PROJECT_ROOT / "data" / "raw" / "reviews.csv"

st.set_page_config(
    page_title="AI Product Recommendation & Review Analysis",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------
# CUSTOM STYLING
# ----------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        padding: 1.75rem 2rem;
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 1.9rem;
        margin-bottom: 0.25rem;
        font-weight: 700;
    }
    .main-header p {
        color: #9ca3af;
        font-size: 0.95rem;
        margin: 0;
    }
    .badge-row {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.75rem;
    }
    .badge {
        background: rgba(255,255,255,0.08);
        color: #d1d5db;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        border: 1px solid rgba(255,255,255,0.15);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------
# DATA LOADING (cached so it only reads the CSVs once)
# ----------------------------------------------------------
@st.cache_data
def load_customers():
    return pd.read_csv(CUSTOMERS_PATH)


@st.cache_data
def load_products():
    return pd.read_csv(PRODUCTS_PATH)


@st.cache_data
def load_reviews():
    return pd.read_csv(REVIEWS_PATH)


def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


data_load_error = None
try:
    customers_df = load_customers()
    products_df = load_products()
    reviews_df = load_reviews()
except FileNotFoundError as e:
    data_load_error = str(e)
    customers_df = pd.DataFrame()
    products_df = pd.DataFrame()
    reviews_df = pd.DataFrame()


# ----------------------------------------------------------
# HEADER
# ----------------------------------------------------------
api_status = check_api_health()
status_badge = "🟢 Backend Online" if api_status else "🔴 Backend Offline"

st.markdown(
    f"""
    <div class="main-header">
        <h1>🛍️ AI Product Recommendation & Review Analysis</h1>
        <p>Personalized recommendations, ML-based sentiment classification, and LLM-powered review insights.</p>
        <div class="badge-row">
            <span class="badge">{status_badge}</span>
            <span class="badge">📦 {len(products_df):,} products</span>
            <span class="badge">👥 {len(customers_df):,} customers</span>
            <span class="badge">💬 {len(reviews_df):,} reviews</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if data_load_error:
    st.error(
        f"Could not find project data files. Make sure this app is run "
        f"from inside the project (or its ui/ subfolder). Missing: {data_load_error}"
    )
    st.stop()

if not api_status:
    st.warning(
        f"Backend API is not reachable at {API_BASE_URL}. "
        "Make sure the FastAPI Docker container is running."
    )


# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 Dataset Snapshot")
    st.metric("Total Products", f"{len(products_df):,}")
    st.metric("Total Customers", f"{len(customers_df):,}")
    st.metric("Total Reviews", f"{len(reviews_df):,}")

    st.markdown("---")
    st.markdown("### Sentiment Split")
    sentiment_counts = reviews_df["sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

    st.markdown("---")
    st.caption(
        "This UI is the front-end for the capstone project. It calls a "
        "FastAPI backend which runs the recommendation engine, the "
        "registered MLflow sentiment model, and the LLM review analyzer."
    )


tab1, tab2, tab3 = st.tabs(
    ["🎯 Recommendations", "💬 Sentiment Checker", "📊 Review Analysis"]
)


# ============================================================
# TAB 1 — RECOMMENDATIONS
# ============================================================
with tab1:
    st.subheader("Get Product Recommendations for a Customer")
    st.caption("Pick a customer from the dataset — no need to remember their ID.")

    # Build a friendly label for each customer so you don't need to
    # memorize CUST00001-style IDs.
    customers_df["display_label"] = (
        customers_df["customer_id"]
        + "  —  "
        + customers_df["city"]
        + "  ·  "
        + customers_df["preferred_category"]
        + " shopper  ·  "
        + customers_df["membership_tier"]
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_label = st.selectbox(
            "Customer",
            options=customers_df["display_label"],
        )
        customer_id = selected_label.split(" ")[0]

    with col2:
        top_n = st.slider("Number of recommendations", 1, 20, 5)

    # Show a quick profile card for the selected customer
    customer_row = customers_df[customers_df["customer_id"] == customer_id].iloc[0]
    profile_cols = st.columns(4)
    profile_cols[0].metric("City", customer_row["city"])
    profile_cols[1].metric("Preferred Category", customer_row["preferred_category"])
    profile_cols[2].metric("Segment", customer_row["customer_segment"])
    profile_cols[3].metric("Membership", customer_row["membership_tier"])

    if st.button("🎯 Get Recommendations", type="primary"):
        with st.spinner("Fetching recommendations..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/recommend",
                    json={"customer_id": customer_id, "top_n": top_n},
                    timeout=10,
                )

                if response.status_code == 200:
                    data = response.json()
                    recs = data["recommendations"]

                    if recs:
                        df = pd.DataFrame(recs)
                        df["recommendation_score"] = df["recommendation_score"].round(3)
                        df = df.rename(
                            columns={
                                "product_id": "Product ID",
                                "product_name": "Product Name",
                                "category": "Category",
                                "recommendation_score": "Score",
                            }
                        )

                        st.success(f"Top {len(recs)} recommendations for {customer_id}")
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        st.bar_chart(df.set_index("Product Name")["Score"])
                    else:
                        st.info("No recommendations found for this customer.")

                elif response.status_code == 404:
                    st.error(f"Customer '{customer_id}' was not found.")
                else:
                    st.error(f"Something went wrong: {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the backend: {e}")


# ============================================================
# TAB 2 — SENTIMENT CHECKER
# ============================================================
with tab2:
    st.subheader("Check the Sentiment of a Review")
    st.caption(
        "Pick a real review from the dataset, or write your own to test the model."
    )

    mode = st.radio(
        "Review source",
        ["Pick an existing review", "Write my own"],
        horizontal=True,
    )

    if mode == "Pick an existing review":
        products_df["display_label"] = (
            products_df["product_id"] + "  —  " + products_df["product_name"]
        )
        selected_product_label = st.selectbox(
            "Product", options=products_df["display_label"], key="sentiment_product"
        )
        selected_product_id = selected_product_label.split(" ")[0]

        product_reviews = reviews_df[reviews_df["product_id"] == selected_product_id]

        if product_reviews.empty:
            st.info("No reviews found for this product in the dataset.")
            review_text = ""
        else:
            review_choice = st.selectbox(
                "Review",
                options=product_reviews["review_text"],
            )
            review_text = review_choice
            actual_label = product_reviews[
                product_reviews["review_text"] == review_choice
            ]["sentiment"].iloc[0]
            st.caption(f"📋 Dataset's original label for this review: **{actual_label}**")
    else:
        review_text = st.text_area(
            "Review text",
            placeholder="e.g. The product quality is excellent and delivery was fast.",
            height=120,
        )

    if st.button("💬 Predict Sentiment", type="primary"):
        if not review_text.strip():
            st.warning("Please select or enter some review text.")
        else:
            with st.spinner("Analyzing sentiment..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/sentiment",
                        json={"review_text": review_text.strip()},
                        timeout=10,
                    )

                    if response.status_code == 200:
                        sentiment = response.json()["sentiment"]
                        colors = {
                            "positive": ("🟢", "green"),
                            "neutral": ("🟡", "orange"),
                            "negative": ("🔴", "red"),
                        }
                        emoji, color = colors.get(sentiment, ("⚪", "gray"))
                        st.markdown(
                            f"### {emoji} Model prediction: :{color}[{sentiment.upper()}]"
                        )
                    else:
                        st.error(f"Something went wrong: {response.text}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Could not reach the backend: {e}")


# ============================================================
# TAB 3 — REVIEW ANALYSIS (LLM)
# ============================================================
with tab3:
    st.subheader("LLM-Powered Review Summary")
    st.caption(
        "Pick a product — we'll automatically pull its real reviews from the "
        "dataset and ask the LLM to summarize praises, complaints, and insights."
    )

    products_df["display_label"] = (
        products_df["product_id"] + "  —  " + products_df["product_name"]
    )
    selected_label = st.selectbox(
        "Product", options=products_df["display_label"], key="analysis_product"
    )
    selected_product_id = selected_label.split(" ")[0]
    selected_product_name = products_df[
        products_df["product_id"] == selected_product_id
    ]["product_name"].iloc[0]

    product_reviews = reviews_df[reviews_df["product_id"] == selected_product_id]
    max_reviews = min(len(product_reviews), 50)

    if product_reviews.empty:
        st.info("No reviews found for this product in the dataset.")
    else:
        review_limit = st.slider(
            "Number of reviews to analyze",
            1,
            max_reviews,
            min(10, max_reviews),
        )

        with st.expander(f"See the {review_limit} reviews that will be analyzed"):
            st.dataframe(
                product_reviews[["sentiment", "review_text"]].head(review_limit).rename(
                    columns={"sentiment": "Sentiment", "review_text": "Review"}
                ),
                use_container_width=True,
                hide_index=True,
            )

        if st.button("📊 Analyze Reviews", type="primary"):
            with st.spinner(f"Analyzing {review_limit} reviews with the LLM..."):
                try:
                    reviews_payload = (
                        product_reviews[["sentiment", "review_text"]]
                        .head(review_limit)
                        .to_dict(orient="records")
                    )

                    analysis_response = requests.post(
                        f"{API_BASE_URL}/review-analysis",
                        json={
                            "product_name": selected_product_name,
                            "reviews": reviews_payload,
                        },
                        timeout=120,
                    )

                    if analysis_response.status_code == 200:
                        result = analysis_response.json()
                        st.success("Analysis complete")

                        st.markdown("### 📝 Overall Summary")
                        st.info(result["summary"])

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("### ✅ Praised Features")
                            if result["praised_features"]:
                                for item in result["praised_features"]:
                                    st.markdown(f"- {item}")
                            else:
                                st.caption("None identified.")

                        with col2:
                            st.markdown("### ⚠️ Common Complaints")
                            if result["common_complaints"]:
                                for item in result["common_complaints"]:
                                    st.markdown(f"- {item}")
                            else:
                                st.caption("None identified.")

                        st.markdown("### 💡 Business Insights")
                        if result["business_insights"]:
                            for item in result["business_insights"]:
                                st.markdown(f"- {item}")
                        else:
                            st.caption("None identified.")

                    else:
                        st.error(f"Something went wrong: {analysis_response.text}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Could not reach the backend: {e}")
