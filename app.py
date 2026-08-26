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

import os

import streamlit as st
import requests
import pandas as pd
from pathlib import Path
import base64
# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8001").rstrip("/")

PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

CUSTOMERS_PATH = PROJECT_ROOT / "data" / "processed" / "recommendation" / "customers.csv"
PRODUCTS_PATH = PROJECT_ROOT / "data" / "processed" / "recommendation" / "products.csv"
REVIEWS_PATH = PROJECT_ROOT / "data" / "raw" / "reviews.csv"
IMAGE_PATH = PROJECT_ROOT / "image.png"
LOGO_PATH = PROJECT_ROOT / "logo1.png"

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Background image not found: {IMAGE_PATH}. "
        "Put image.png in the same folder as app.py."
    )

if not LOGO_PATH.exists():
    raise FileNotFoundError(
        f"Logo image not found: {LOGO_PATH}. "
        "Put logo1.png in the same folder as app.py."
    )

with open(IMAGE_PATH, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

with open(LOGO_PATH, "rb") as f:
    logo_base64 = base64.b64encode(f.read()).decode("utf-8")

PAGE_ICON = str(LOGO_PATH) if LOGO_PATH.exists() else str(IMAGE_PATH)

st.set_page_config(
    page_title="Product Recommendations & Reviews",
    page_icon=PAGE_ICON,
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
css="""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #070910;
    --ink: #f8fafc;
    --muted: #94a3b8;
    --line: rgba(148,163,184,.15);
    --purple: #7c6cff;
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: #070910 !important;
    color: var(--ink) !important;
    position: relative;
    min-height: 100vh;
    isolation: isolate;
}

/* Keep every Streamlit surface transparent so the image can actually show. */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
[data-testid="stAppViewContainer"] > .main > div {
    background: transparent !important;
}

/* Full-screen faded background image. */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background:
        linear-gradient(
            180deg,
            rgba(4, 6, 12, .22),
            rgba(4, 6, 12, .42)
        ),
        url("data:image/png;base64,image.png");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    opacity: .82;
    z-index: -2;
    pointer-events: none;
}

/* Very subtle dark tint: keeps text readable without washing out the image. */
.stApp::after {
    content: "";
    position: fixed;
    inset: 0;
    background:
        radial-gradient(
            circle at 50% 42%,
            rgba(99, 102, 241, .05),
            transparent 30%
        ),
        linear-gradient(
            180deg,
            rgba(5, 7, 14, .12),
            rgba(5, 7, 14, .28)
        );
    z-index: -1;
    pointer-events: none;
}

.block-container {
    max-width: 1180px !important;
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    position: relative;
    z-index: 3;
}

#MainMenu,
footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    visibility: visible !important;
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
    border: none !important;
}
header[data-testid="stHeader"] > div {
    background: transparent !important;
}
header[data-testid="stHeader"] button {
    background: transparent !important;
    color: #f8fafc !important;
}

/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        rgba(7, 12, 23, 0.97),
        rgba(9, 17, 31, 0.96)
    ) !important;

    border-right: 1px solid rgba(148, 163, 184, 0.12) !important;
}

/* Keep Streamlit's sidebar controls clickable */
section[data-testid="stSidebar"] > div {
    background: transparent !important;
}

/* Sidebar content */
section[data-testid="stSidebar"] * {
    color: #e8edf7;
}

/* IMPORTANT:
   Do not force z-index on the sidebar itself.
   Streamlit controls need to remain above/beside it.
*/
[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999 !important;
}

/* Sidebar expand button */
button[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
}

/* Sidebar close button */
button[data-testid="stSidebarCollapseButton"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
}

.sidebar-brand { padding: .15rem 0 1.3rem; }
section[data-testid="stSidebar"] [data-testid="stImage"] {
    margin-bottom: .35rem !important;
}
section[data-testid="stSidebar"] [data-testid="stImage"] img {
    width: 55px !important;
    height: 55px !important;
    object-fit: contain !important;
    display: block !important;
}

.sidebar-brand h3 {
    color: #fff !important;
    font-size: 1rem;
    margin: 0;
    font-weight: 750;
}

.sidebar-brand p {
    color: #94a3b8 !important;
    font-size: .78rem;
    margin: .25rem 0 0;
    line-height: 1.45;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: .45rem;
    padding: .42rem .7rem;
    border-radius: 999px;
    background: rgba(34,197,94,.10);
    border: 1px solid rgba(74,222,128,.25);
    color: #bbf7d0 !important;
    font-size: .75rem;
    font-weight: 700;
}

.status-dot {
    width: 7px;
    height: 7px;
    background: #4ade80;
    border-radius: 50%;
    display: inline-block;
}

/* Top bar */
.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1rem;
}

.brand {
    display: flex;
    align-items: center;
    gap: .75rem;
}

.brand-mark {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    box-shadow: 0 8px 28px rgba(99,102,241,.28);
    font-size: 1.35rem;
}

.brand-name {
    color: #f8fafc !important;
    font-size: 1rem;
    font-weight: 800;
}

.brand-sub {
    color: #94a3b8 !important;
    font-size: .74rem;
    margin-top: .08rem;
}

.top-status {
    display: inline-flex;
    align-items: center;
    gap: .45rem;
    background: rgba(10,15,28,.72);
    border: 1px solid rgba(148,163,184,.18);
    border-radius: 999px;
    padding: .45rem .75rem;
    color: #bbf7d0 !important;
    font-size: .73rem;
    font-weight: 700;
    backdrop-filter: blur(12px);
}

/* Hero */
.hero {
    position: relative;
    overflow: hidden;
    padding: 2.2rem 2.5rem;
    border-radius: 22px;
    margin: .35rem 0 1.4rem;
    background:
        radial-gradient(circle at 90% 15%, rgba(124,108,255,.22), transparent 28%),
        linear-gradient(135deg, rgba(11,27,48,.72), rgba(19,51,82,.62));
    border: 1px solid rgba(96,165,250,.18);
    box-shadow: 0 22px 55px rgba(0,0,0,.25);
}

.hero:after {
    content: "";
    position: absolute;
    width: 220px;
    height: 220px;
    right: -90px;
    bottom: -110px;
    border-radius: 50%;
    border: 1px solid rgba(255,255,255,.12);
}

.hero-kicker, .section-label {
    color: #818cf8 !important;
    text-transform: uppercase;
    letter-spacing: .1em;
    font-size: .68rem;
    font-weight: 800;
}

.hero-kicker { margin-bottom: .55rem; }

.hero h1 {
    color: #fff !important;
    font-size: 2.15rem;
    line-height: 1.08;
    font-weight: 800;
    margin: 0 0 .7rem;
}

.hero p {
    color: #dbeafe !important;
    font-size: .96rem;
    line-height: 1.55;
    margin: 0;
    max-width: 720px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: .25rem;
    background: rgba(8,12,22,.66);
    backdrop-filter: blur(14px);
    border-bottom: 1px solid rgba(148,163,184,.14);
    padding: .25rem;
}

.stTabs [data-baseweb="tab"] {
    height: 42px !important;
    padding: 0 1.1rem !important;
    border-radius: 10px !important;
    color: #cbd5e1 !important;
    font-size: .84rem !important;
    font-weight: 700 !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #fff !important;
    background: rgba(124,108,255,.10) !important;
}

.stTabs [aria-selected="true"] {
    color: #fff !important;
    background: rgba(124,108,255,.13) !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background: linear-gradient(90deg,#7c6cff,#38bdf8) !important;
    height: 3px !important;
    border-radius: 999px !important;
}

.section-label { margin: 1.55rem 0 .7rem; }

.section-title {
    color: #f8fafc !important;
    font-size: 1.25rem;
    font-weight: 800;
    margin: 0;
}

.section-subtitle {
    color: #94a3b8 !important;
    font-size: .82rem;
    margin: .2rem 0 1rem;
}

/* Controls */
div[data-testid="stSelectbox"] label,
div[data-testid="stTextArea"] label,
div[data-testid="stRadio"] label {
    color: #94a3b8 !important;
    font-weight: 700 !important;
    font-size: .8rem !important;
}

div[data-baseweb="select"] > div {
    background: rgba(18,22,35,.66) !important;
    border: 1px solid rgba(148,163,184,.16) !important;
    border-radius: 10px !important;
    min-height: 44px !important;
}

div[data-baseweb="select"] * { color: #f8fafc !important; }

textarea {
    background: rgba(18,22,35,.66) !important;
    color: #f8fafc !important;
    border: 1px solid rgba(148,163,184,.16) !important;
    border-radius: 12px !important;
}

textarea::placeholder { color: #64748b !important; }

.stButton > button {
    border-radius: 10px !important;
    font-weight: 750 !important;
    min-height: 42px !important;
    border: 1px solid rgba(148,163,184,.16) !important;
    background: rgba(18,22,35,.82) !important;
    color: #f8fafc !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#635bff,#7c3aed) !important;
    border-color: #7c6cff !important;
    color: #fff !important;
    box-shadow: 0 9px 25px rgba(99,91,255,.28);
}

/* Glass cards */
.profile-card {
    display: grid;
    grid-template-columns: repeat(4,minmax(0,1fr));
    gap: .65rem;
    margin: .8rem 0 1.15rem;
}

.profile-pill, .product-card, .summary-card {
    background: rgba(10,14,25,.56) !important;
    border: 1px solid rgba(148,163,184,.14) !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}

.profile-pill {
    border-radius: 13px;
    padding: .8rem .9rem;
    box-shadow: 0 8px 24px rgba(0,0,0,.14);
}

.profile-pill .label {
    color: #7f91aa !important;
    font-size: .63rem;
    font-weight: 800;
    text-transform: uppercase;
}

.profile-pill .value {
    color: #f8fafc !important;
    font-size: .87rem;
    font-weight: 750;
    margin-top: .18rem;
}

.product-card {
    border-radius: 14px;
    padding: 1rem 1.15rem;
    margin-bottom: .65rem;
    box-shadow: 0 8px 24px rgba(0,0,0,.14);
}

.product-card-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1rem;
}

.product-rank {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    background: rgba(99,91,255,.14);
    color: #a5b4fc !important;
    font-weight: 800;
    border-radius: 8px;
    margin-right: .55rem;
}

.product-name { color: #f8fafc !important; font-size: .96rem; font-weight: 800; }
.product-meta { color: #64748b !important; font-size: .76rem; margin-top: .18rem; }

.category-tag {
    background: rgba(56,189,248,.10);
    color: #7dd3fc !important;
    font-size: .68rem;
    font-weight: 750;
    padding: .25rem .6rem;
    border-radius: 999px;
}

.score-bar-track {
    background: rgba(148,163,184,.12);
    border-radius: 999px;
    height: 7px;
    margin-top: .75rem;
    overflow: hidden;
}

.score-bar-fill {
    background: linear-gradient(90deg,#635bff,#38bdf8);
    height: 100%;
    border-radius: 999px;
}

.score-label { color: #64748b !important; font-size: .7rem; margin-top: .3rem; }

/* Sentiment and review cards */
.sentiment-card {
    border-radius: 16px;
    padding: 1.25rem 1.4rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-top: .8rem;
}

.sentiment-card.positive { background: rgba(22,163,74,.10); border: 1px solid rgba(74,222,128,.22); }
.sentiment-card.neutral { background: rgba(245,158,11,.10); border: 1px solid rgba(251,191,36,.22); }
.sentiment-card.negative { background: rgba(244,63,94,.10); border: 1px solid rgba(251,113,133,.22); }

.sentiment-icon { font-size: 2rem; }
.sentiment-title { font-size: 1.05rem; font-weight: 850; margin: 0; }
.sentiment-title.positive { color: #4ade80 !important; }
.sentiment-title.neutral { color: #fbbf24 !important; }
.sentiment-title.negative { color: #fb7185 !important; }
.sentiment-sub { color: #94a3b8 !important; font-size: .8rem; }

.summary-card {
    border-left: 4px solid #6366f1 !important;
    border-radius: 12px;
    padding: 1.15rem 1.3rem;
    margin: .5rem 0 1rem;
}

.summary-card p { color: #cbd5e1 !important; margin: 0; font-size: .88rem; line-height: 1.6; }

.insight-card {
    border-radius: 13px;
    padding: 1.15rem 1.25rem;
    height: 100%;
}

.insight-card.praise { background: rgba(22,163,74,.09); border: 1px solid rgba(74,222,128,.18); }
.insight-card.complaint { background: rgba(244,63,94,.09); border: 1px solid rgba(251,113,133,.18); }
.insight-card.praise h4 { color: #4ade80 !important; }
.insight-card.complaint h4 { color: #fb7185 !important; }
.insight-list li { color: #cbd5e1 !important; font-size: .8rem; }

.business-card {
    background: linear-gradient(135deg,rgba(30,27,75,.88),rgba(17,45,75,.88));
    border: 1px solid rgba(124,108,255,.18);
    border-radius: 14px;
    padding: 1.15rem 1.3rem;
    margin-top: .8rem;
}

.business-card h4 { color: #a5b4fc !important; }
.business-card li { color: #e2e8f0 !important; font-size: .8rem; }
.chart-caption { color: #64748b !important; font-size: .7rem; text-align: center; }

[data-testid="stDataFrame"] {
    border: 1px solid rgba(148,163,184,.14) !important;
}

@media (max-width: 800px) {
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    .hero { padding: 1.6rem; }
    .hero h1 { font-size: 1.7rem; }
    .profile-card { grid-template-columns: repeat(2,minmax(0,1fr)); }
}
</style>
"""
css = css.replace("image.png", image_base64)

st.markdown(
    css,
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
    f"""
    <div class="topbar">
        <div class="brand">
            <div class="brand-mark">
                <img src="data:image/png;base64,{logo_base64}" alt="Product Intelligence">
            </div>
            <div>
                <div class="brand-name">VOID - Your Personal Shopping Intelligence</div>
                <div class="brand-sub">Recommendations · Sentiment · Review Insights</div>
            </div>
        </div>
        <div class="top-status"><span>●</span> API ready</div>
    </div>

    <div class="hero">
        <div class="hero-kicker">Personalized shopping intelligence</div>
        <h1>Discover products you'll love.</h1>
        <p>
            Get personalized recommendations, understand customer sentiment,
            and turn product reviews into clear, actionable insights.
        </p>
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
    st.image(str(LOGO_PATH), width=55)

    st.markdown(
        """
        <div class="sidebar-brand">
            <h3>Product Intelligence</h3>
            <p>Customer recommendations and review intelligence in one place.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="status-pill"><span class="status-dot"></span> {"API connected" if api_status else "API unavailable"}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### Store snapshot")
    m1, m2 = st.columns(2)
    m1.metric("Products", f"{len(products_df):,}")
    m2.metric("Categories", f"{products_df['category'].nunique()}")
    m1.metric("Reviews", f"{len(reviews_df):,}")
    m2.metric("Happy Customers", f"{(reviews_df['sentiment'] == 'positive').sum():,}")

    st.markdown("---")
    st.markdown(
        f'<div style="color:#9fb4c9;font-size:.72rem;margin-top:.6rem;">'
        f'System status: {"available" if api_status else "not available"}</div>',
        unsafe_allow_html=True,
    )


tab1, tab2, tab3 = st.tabs(
    ["  Recommendations", "  Sentiment Checker", "  Review Analysis"]
)


# ============================================================
# TAB 1 — RECOMMENDATIONS
# ============================================================
with tab1:
    st.markdown('<div class="section-label">Personalized recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Find the right products</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Filter shoppers by profile, then generate their strongest product matches.</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="section-label">AI sentiment checker</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">How does this review feel?</div>', unsafe_allow_html=True)
    st.caption("Type any review below and we'll tell you whether it reads as positive, neutral, or negative.")

    if "review_text_value" not in st.session_state:
        st.session_state["review_text_value"] = ""

    review_text = st.text_area(
        "Your review",
        placeholder="e.g. The product quality is excellent and delivery was fast.",
        height=110,
        label_visibility="collapsed",
        value=st.session_state["review_text_value"],
        key="review_text_input",
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
                st.session_state["review_text_value"] = ex_review_choice
                st.rerun()
        else:
            st.caption("No reviews available for this product yet.")

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
    st.markdown('<div class="section-label">Review intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Understand what customers think</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Choose a product and let the review analysis engine surface the themes that matter.</div>', unsafe_allow_html=True)

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
                            sentiment_counts = (
                                analysis_slice["sentiment"]
                                .value_counts()
                                .reindex(["positive", "neutral", "negative"])
                                .dropna()
                                .astype(int)
                            )
                            st.bar_chart(sentiment_counts)
                            st.markdown(
                                '<p class="chart-caption">How customers feel overall</p>',
                                unsafe_allow_html=True,
                            )

                        with chart_col2:
                            rating_counts = (
                                analysis_slice["rating"]
                                .value_counts()
                                .sort_index()
                                .astype(int)
                            )
                            st.bar_chart(rating_counts)
                            st.markdown(
                                '<p class="chart-caption">Star rating breakdown</p>',
                                unsafe_allow_html=True,
                            )

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