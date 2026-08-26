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
"""
==========================================================
STREAMLIT UI (v4) — Customer-Facing Version
==========================================================

WHAT CHANGED FROM v3, based on feedback:
-----------------------------------------
1. Removed "Capstone Project" label and all backend/tech jargon from
   the main screen — this now reads like a real customer product,
   not a student project. Backend status moved to a quiet sidebar
   note instead of a big chip up top.
2. Recommendations tab: added City + Preferred Category filters
   before picking a customer, so narrowing down feels natural.
3. Sentiment Checker: now defaults to "write your own review" as
   the primary way to use it (realistic for a customer). Trying an
   existing example is a secondary option, and product selection for
   examples uses a two-step Category -> Product cascading dropdown.
4. Review Analysis: product selection is also Category -> Product
   cascading (Beauty separate, Electronics separate, etc.), the
   "number of reviews" slider was replaced with clear preset choices
   (Quick / Standard / Full), and results now include two charts
   (sentiment split + star rating split) so a non-technical viewer
   can understand the analysis at a glance.

HOW TO RUN:
    Terminal 1: uvicorn api.main:app --reload
    Terminal 2: streamlit run ui/app.py

NOTE: this version uses Plotly for charts. If not already installed:
    pip install plotly
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
API_BASE_URL = "http://localhost:9000"

PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

CUSTOMERS_PATH = PROJECT_ROOT / "data" / "processed" / "recommendation" / "customers.csv"
PRODUCTS_PATH = PROJECT_ROOT / "data" / "processed" / "recommendation" / "products.csv"
REVIEWS_PATH = PROJECT_ROOT / "data" / "raw" / "reviews.csv"

st.set_page_config(
    page_title="Product Recommendations & Reviews",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "positive": "#16A34A",
    "neutral": "#D97706",
    "negative": "#DC2626",
}


# ----------------------------------------------------------
# STYLING
# ----------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    :root {
        --navy: #0B1F3A;
        --navy-light: #16305C;
        --accent: #2563EB;
        --accent-light: #EFF4FF;
        --green: #16A34A;
        --green-light: #F0FDF4;
        --amber: #D97706;
        --amber-light: #FFFBEB;
        --red: #DC2626;
        --red-light: #FEF2F2;
        --gray-50: #F8FAFC;
        --gray-100: #F1F5F9;
        --gray-500: #64748B;
        --gray-700: #334155;
        --gray-900: #0F172A;
    }
    .stApp { background-color: var(--gray-50); }

    .hero {
        padding: 2.25rem 2.5rem;
        background: linear-gradient(135deg, var(--navy) 0%, var(--navy-light) 100%);
        border-radius: 16px;
        margin-bottom: 1.75rem;
        box-shadow: 0 10px 30px -10px rgba(11, 31, 58, 0.4);
    }
    .hero h1 { color: #ffffff; font-size: 2rem; font-weight: 800; margin: 0 0 0.4rem 0; letter-spacing: -0.02em; }
    .hero p { color: #CBD5E1; font-size: 1rem; margin: 0; max-width: 640px; }

    .section-label {
        font-size: 0.78rem; font-weight: 700; letter-spacing: 0.06em;
        text-transform: uppercase; color: var(--gray-500); margin: 1.4rem 0 0.6rem 0;
    }

    .profile-card { display: flex; gap: 0.75rem; flex-wrap: wrap; margin: 0.75rem 0 1.25rem 0; }
    .profile-pill { background: var(--gray-100); border-radius: 10px; padding: 0.6rem 1rem; min-width: 130px; }
    .profile-pill .label { font-size: 0.7rem; color: var(--gray-500); font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
    .profile-pill .value { font-size: 0.95rem; color: var(--gray-900); font-weight: 600; margin-top: 0.1rem; }

    .product-card {
        background: #ffffff; border: 1px solid var(--gray-100); border-radius: 14px;
        padding: 1.1rem 1.3rem; margin-bottom: 0.7rem; box-shadow: 0 1px 3px rgba(15,23,42,0.04);
    }
    .product-card-top { display: flex; justify-content: space-between; align-items: flex-start; }
    .product-rank {
        display: inline-flex; align-items: center; justify-content: center;
        width: 26px; height: 26px; background: var(--accent-light); color: var(--accent);
        font-weight: 700; font-size: 0.8rem; border-radius: 8px; margin-right: 0.6rem;
    }
    .product-name { font-size: 1.02rem; font-weight: 700; color: var(--gray-900); }
    .product-meta { color: var(--gray-500); font-size: 0.83rem; margin-top: 0.15rem; }
    .category-tag { background: var(--accent-light); color: var(--accent); font-size: 0.72rem; font-weight: 600; padding: 0.2rem 0.6rem; border-radius: 999px; }
    .score-bar-track { background: var(--gray-100); border-radius: 999px; height: 7px; margin-top: 0.7rem; overflow: hidden; }
    .score-bar-fill { background: linear-gradient(90deg, var(--accent) 0%, #60A5FA 100%); height: 100%; border-radius: 999px; }
    .score-label { font-size: 0.75rem; color: var(--gray-500); margin-top: 0.35rem; }

    .sentiment-card { border-radius: 16px; padding: 1.6rem 1.8rem; display: flex; align-items: center; gap: 1.1rem; margin-top: 0.5rem; }
    .sentiment-card.positive { background: var(--green-light); border: 1px solid #BBF7D0; }
    .sentiment-card.neutral { background: var(--amber-light); border: 1px solid #FDE68A; }
    .sentiment-card.negative { background: var(--red-light); border: 1px solid #FECACA; }
    .sentiment-icon { font-size: 2.2rem; }
    .sentiment-title { font-size: 1.3rem; font-weight: 800; margin: 0; }
    .sentiment-title.positive { color: var(--green); }
    .sentiment-title.neutral { color: var(--amber); }
    .sentiment-title.negative { color: var(--red); }
    .sentiment-sub { color: var(--gray-500); font-size: 0.87rem; margin-top: 0.1rem; }

    .insight-card { border-radius: 14px; padding: 1.3rem 1.4rem; height: 100%; }
    .insight-card.praise { background: var(--green-light); border: 1px solid #BBF7D0; }
    .insight-card.complaint { background: var(--red-light); border: 1px solid #FECACA; }
    .insight-card h4 { margin: 0 0 0.7rem 0; font-size: 0.95rem; font-weight: 700; }
    .insight-card.praise h4 { color: #15803D; }
    .insight-card.complaint h4 { color: #B91C1C; }
    .insight-list { margin: 0; padding-left: 1.1rem; }
    .insight-list li { font-size: 0.88rem; color: var(--gray-700); margin-bottom: 0.35rem; line-height: 1.4; }

    .summary-card { background: var(--accent-light); border: 1px solid #BFDBFE; border-radius: 14px; padding: 1.3rem 1.5rem; margin: 0.5rem 0 1rem 0; }
    .summary-card p { margin: 0; color: var(--gray-900); font-size: 0.95rem; line-height: 1.55; }

    .business-card { background: var(--navy); border-radius: 14px; padding: 1.3rem 1.5rem; margin-top: 0.9rem; }
    .business-card h4 { color: #93C5FD; margin: 0 0 0.7rem 0; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }
    .business-card ol { margin: 0; padding-left: 1.2rem; }
    .business-card li { color: #E2E8F0; font-size: 0.9rem; margin-bottom: 0.4rem; line-height: 1.5; }

    .stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; padding: 0.6rem 1.3rem; border-radius: 10px 10px 0 0; }
    .stButton button { border-radius: 10px; font-weight: 600; padding: 0.55rem 1.4rem; }

    div[data-testid="stRadio"] > label { font-weight: 600; }

    section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid var(--gray-100); }
    .chart-caption { color: var(--gray-500); font-size: 0.82rem; text-align: center; margin-top: -0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------
# DATA LOADING
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
# HERO HEADER — customer-facing, no tech jargon
# ----------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🛍️ Discover Products You'll Love</h1>
        <p>Personalized recommendations picked for you, plus real insights from
        thousands of customer reviews — so you always know what to expect.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if data_load_error:
    st.error(f"Could not load product data. Missing: {data_load_error}")
    st.stop()

api_status = check_api_health()
if not api_status:
    st.warning("The recommendation service is temporarily unavailable. Please try again shortly.")


# ----------------------------------------------------------
# SIDEBAR — kept light, non-technical
# ----------------------------------------------------------
with st.sidebar:
    st.markdown("#### 🛒 Store Snapshot")
    m1, m2 = st.columns(2)
    m1.metric("Products", f"{len(products_df):,}")
    m2.metric("Categories", f"{products_df['category'].nunique()}")
    m1.metric("Reviews", f"{len(reviews_df):,}")
    m2.metric("Happy Customers", f"{(reviews_df['sentiment'] == 'positive').sum():,}")

    st.markdown("---")
    st.caption("Product Recommendations & Review Insights")
    st.caption(f"System status: {'🟢 Available' if api_status else '🔴 Unavailable'}")


tab1, tab2, tab3 = st.tabs(
    ["🎯  Recommendations", "💬  Sentiment Checker", "📊  Review Analysis"]
)


# ============================================================
# TAB 1 — RECOMMENDATIONS
# ============================================================
with tab1:
    st.markdown('<div class="section-label">Find recommendations for a shopper</div>', unsafe_allow_html=True)

    f1, f2 = st.columns(2)
    with f1:
        city_options = ["All Cities"] + sorted(customers_df["city"].unique().tolist())
        selected_city = st.selectbox("Filter by city", city_options)
    with f2:
        category_options = ["All Categories"] + sorted(customers_df["preferred_category"].unique().tolist())
        selected_pref_category = st.selectbox("Filter by preferred category", category_options)

    filtered_customers = customers_df.copy()
    if selected_city != "All Cities":
        filtered_customers = filtered_customers[filtered_customers["city"] == selected_city]
    if selected_pref_category != "All Categories":
        filtered_customers = filtered_customers[filtered_customers["preferred_category"] == selected_pref_category]

    if filtered_customers.empty:
        st.info("No shoppers match this filter combination. Try widening your filters.")
    else:
        filtered_customers = filtered_customers.copy()
        filtered_customers["display_label"] = (
            filtered_customers["customer_id"]
            + "  —  "
            + filtered_customers["city"]
            + "  ·  "
            + filtered_customers["preferred_category"]
            + " shopper"
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            selected_label = st.selectbox("Shopper", options=filtered_customers["display_label"])
            customer_id = selected_label.split(" ")[0]
        with col2:
            top_n = st.selectbox("Show", options=[3, 5, 10, 15], index=1, format_func=lambda n: f"Top {n}")

        customer_row = filtered_customers[filtered_customers["customer_id"] == customer_id].iloc[0]
        st.markdown(
            f"""
            <div class="profile-card">
                <div class="profile-pill"><div class="label">City</div><div class="value">{customer_row['city']}</div></div>
                <div class="profile-pill"><div class="label">Prefers</div><div class="value">{customer_row['preferred_category']}</div></div>
                <div class="profile-pill"><div class="label">Segment</div><div class="value">{customer_row['customer_segment']}</div></div>
                <div class="profile-pill"><div class="label">Membership</div><div class="value">{customer_row['membership_tier']}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Get Recommendations", type="primary"):
            with st.spinner("Finding the best matches..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/recommend",
                        json={"customer_id": customer_id, "top_n": top_n},
                        timeout=10,
                    )

                    if response.status_code == 200:
                        recs = response.json()["recommendations"]

                        if recs:
                            st.markdown(
                                f'<div class="section-label">Top {len(recs)} recommended products</div>',
                                unsafe_allow_html=True,
                            )
                            max_score = max(r["recommendation_score"] for r in recs) or 1

                            for i, r in enumerate(recs, start=1):
                                pct = round((r["recommendation_score"] / max_score) * 100, 1)
                                st.markdown(
                                    f"""
                                    <div class="product-card">
                                        <div class="product-card-top">
                                            <div>
                                                <span class="product-rank">{i}</span>
                                                <span class="product-name">{r['product_name']}</span>
                                                <div class="product-meta">{r['product_id']}</div>
                                            </div>
                                            <span class="category-tag">{r['category']}</span>
                                        </div>
                                        <div class="score-bar-track"><div class="score-bar-fill" style="width:{pct}%;"></div></div>
                                        <div class="score-label">Match strength: {pct}%</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.info("No recommendations found for this shopper yet.")

                    elif response.status_code == 404:
                        st.error("Shopper not found.")
                    else:
                        st.error("Something went wrong. Please try again.")

                except requests.exceptions.RequestException:
                    st.error("The recommendation service is unavailable right now.")


# ============================================================
# TAB 2 — SENTIMENT CHECKER
# ============================================================
with tab2:
    st.markdown('<div class="section-label">Check how a review sounds</div>', unsafe_allow_html=True)
    st.caption("Type any review below and we'll tell you whether it reads as positive, neutral, or negative.")

    review_text = st.text_area(
        "Your review",
        placeholder="e.g. The product quality is excellent and delivery was fast.",
        height=110,
        label_visibility="collapsed",
    )

    with st.expander("Not sure what to try? Load a real example review instead"):
        cat_options = sorted(products_df["category"].unique().tolist())
        ex_cat = st.selectbox("Category", options=cat_options, key="sentiment_cat")

        cat_products = products_df[products_df["category"] == ex_cat].copy()
        cat_products["display_label"] = cat_products["product_id"] + "  —  " + cat_products["product_name"]
        ex_product_label = st.selectbox("Product", options=cat_products["display_label"], key="sentiment_prod")
        ex_product_id = ex_product_label.split(" ")[0]

        ex_reviews = reviews_df[reviews_df["product_id"] == ex_product_id]
        if not ex_reviews.empty:
            ex_review_choice = st.selectbox("Example review", options=ex_reviews["review_text"], key="sentiment_review_pick")
            if st.button("Use this example"):
                st.session_state["prefill_review"] = ex_review_choice
                st.rerun()
        else:
            st.caption("No reviews available for this product yet.")

    if "prefill_review" in st.session_state and not review_text:
        review_text = st.session_state["prefill_review"]
        st.info(f"Loaded example: \u201c{review_text[:120]}{'...' if len(review_text) > 120 else ''}\u201d")

    if st.button("Predict Sentiment", type="primary"):
        if not review_text.strip():
            st.warning("Please write or load a review first.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/sentiment",
                        json={"review_text": review_text.strip()},
                        timeout=10,
                    )

                    if response.status_code == 200:
                        sentiment = response.json()["sentiment"]
                        icons = {"positive": "😊", "neutral": "😐", "negative": "😞"}
                        icon = icons.get(sentiment, "🤔")
                        messages = {
                            "positive": "This review sounds happy and satisfied.",
                            "neutral": "This review sounds balanced or mixed.",
                            "negative": "This review sounds dissatisfied.",
                        }

                        st.markdown(
                            f"""
                            <div class="sentiment-card {sentiment}">
                                <div class="sentiment-icon">{icon}</div>
                                <div>
                                    <p class="sentiment-title {sentiment}">{sentiment.upper()}</p>
                                    <p class="sentiment-sub">{messages.get(sentiment, "")}</p>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.error("Something went wrong. Please try again.")

                except requests.exceptions.RequestException:
                    st.error("The sentiment service is unavailable right now.")


# ============================================================
# TAB 3 — REVIEW ANALYSIS (LLM)
# ============================================================
with tab3:
    st.markdown('<div class="section-label">Understand what customers think about a product</div>', unsafe_allow_html=True)

    cat_options = sorted(products_df["category"].unique().tolist())
    selected_category = st.selectbox("1. Choose a category", options=cat_options, key="analysis_cat")

    cat_products = products_df[products_df["category"] == selected_category].copy()
    cat_products["display_label"] = cat_products["product_id"] + "  —  " + cat_products["product_name"]
    selected_label = st.selectbox("2. Choose a product", options=cat_products["display_label"], key="analysis_product")
    selected_product_id = selected_label.split(" ")[0]
    selected_product_name = cat_products[cat_products["product_id"] == selected_product_id]["product_name"].iloc[0]

    product_reviews = reviews_df[reviews_df["product_id"] == selected_product_id]
    total_available = len(product_reviews)

    if product_reviews.empty:
        st.info("No reviews found for this product yet.")
    else:
        st.markdown('<div class="section-label">3. Choose analysis depth</div>', unsafe_allow_html=True)

        depth_options = []
        if total_available >= 10:
            depth_options.append(("Quick scan", min(10, total_available)))
        if total_available >= 25:
            depth_options.append(("Standard", min(25, total_available)))
        depth_options.append(("Full analysis", total_available))

        seen_counts = set()
        depth_options_clean = []
        for label, count in depth_options:
            if count not in seen_counts:
                depth_options_clean.append((label, count))
                seen_counts.add(count)

        depth_labels = [f"{label} ({count} reviews)" for label, count in depth_options_clean]
        chosen_depth_label = st.radio("Analysis depth", depth_labels, horizontal=True, label_visibility="collapsed")
        review_limit = dict(zip(depth_labels, [c for _, c in depth_options_clean]))[chosen_depth_label]

        with st.expander(f"View the {review_limit} reviews being analyzed"):
            st.dataframe(
                product_reviews[["rating", "sentiment", "review_text"]].head(review_limit).rename(
                    columns={"rating": "Stars", "sentiment": "Sentiment", "review_text": "Review"}
                ),
                use_container_width=True,
                hide_index=True,
            )

        if st.button("Analyze Reviews", type="primary"):
            with st.spinner(f"Reading {review_limit} reviews and summarizing..."):
                try:
                    analysis_slice = product_reviews.head(review_limit)
                    reviews_payload = analysis_slice[["sentiment", "review_text"]].to_dict(orient="records")

                    analysis_response = requests.post(
                        f"{API_BASE_URL}/review-analysis",
                        json={"product_name": selected_product_name, "reviews": reviews_payload},
                        timeout=30,
                    )

                    if analysis_response.status_code == 200:
                        result = analysis_response.json()

                        st.markdown('<div class="section-label">At a glance</div>', unsafe_allow_html=True)
                        chart_col1, chart_col2 = st.columns(2)

                        with chart_col1:
                            sentiment_counts = analysis_slice["sentiment"].value_counts()
                            labels = sentiment_counts.index.tolist()
                            values = sentiment_counts.values.tolist()
                            colors = [COLORS.get(l, "#94A3B8") for l in labels]

                            fig_donut = go.Figure(
                                data=[
                                    go.Pie(
                                        labels=[l.capitalize() for l in labels],
                                        values=values,
                                        hole=0.6,
                                        marker=dict(colors=colors),
                                        textinfo="percent",
                                        textfont=dict(size=14),
                                    )
                                ]
                            )
                            fig_donut.update_layout(
                                showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=-0.15),
                                margin=dict(t=10, b=10, l=10, r=10),
                                height=280,
                                annotations=[dict(text="Sentiment", x=0.5, y=0.5, font_size=14, showarrow=False)],
                            )
                            st.plotly_chart(fig_donut, use_container_width=True)
                            st.markdown('<p class="chart-caption">How customers feel overall</p>', unsafe_allow_html=True)

                        with chart_col2:
                            rating_counts = analysis_slice["rating"].value_counts().sort_index()
                            fig_bar = go.Figure(
                                data=[
                                    go.Bar(
                                        x=[f"{r} \u2605" for r in rating_counts.index],
                                        y=rating_counts.values,
                                        marker_color="#2563EB",
                                    )
                                ]
                            )
                            fig_bar.update_layout(
                                margin=dict(t=10, b=10, l=10, r=10),
                                height=280,
                                yaxis_title=None,
                                xaxis_title=None,
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)
                            st.markdown('<p class="chart-caption">Star rating breakdown</p>', unsafe_allow_html=True)

                        st.markdown('<div class="section-label">Summary</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="summary-card"><p>{result["summary"]}</p></div>', unsafe_allow_html=True)

                        col1, col2 = st.columns(2)
                        with col1:
                            praises = "".join(f"<li>{item}</li>" for item in result["praised_features"]) or "<li>None identified.</li>"
                            st.markdown(
                                f'<div class="insight-card praise"><h4>✅ What Customers Love</h4><ul class="insight-list">{praises}</ul></div>',
                                unsafe_allow_html=True,
                            )
                        with col2:
                            complaints = "".join(f"<li>{item}</li>" for item in result["common_complaints"]) or "<li>None identified.</li>"
                            st.markdown(
                                f'<div class="insight-card complaint"><h4>⚠️ Common Concerns</h4><ul class="insight-list">{complaints}</ul></div>',
                                unsafe_allow_html=True,
                            )

                        insights = "".join(f"<li>{item}</li>" for item in result["business_insights"]) or "<li>None identified.</li>"
                        st.markdown(
                            f'<div class="business-card"><h4>💡 Key Takeaways</h4><ol>{insights}</ol></div>',
                            unsafe_allow_html=True,
                        )

                    else:
                        st.error("Something went wrong. Please try again.")

                except requests.exceptions.RequestException:
                    st.error("The analysis service is unavailable right now.")