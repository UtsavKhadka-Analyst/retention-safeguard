"""
=============================================================================
PHASE 4 | app.py — Streamlit Dashboard
=============================================================================
Proactive Retention & Revenue Safeguard System
Author: Utsav Khadka (github.com/UtsavKhadka-Analyst)

PURPOSE:
    A marketing-manager-facing dashboard that connects to the FastAPI
    backend (Phase 3) and provides actionable retention insights.

TARGET USER:
    Non-technical marketing manager who needs to:
    1. See total revenue at risk from churn
    2. Identify which customers to target with retention offers
    3. Allocate a limited marketing budget for maximum ROI
    4. Download a targeted customer list for campaign execution

DESIGN PHILOSOPHY:
    - Lead with BUSINESS METRICS, not model metrics
    - Every number should answer "so what?" for a non-technical user
    - The dashboard makes DECISIONS easier, not just displays data
    - No jargon: "Persuadable customers" not "positive CATE segment"


=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import sys

# ── Add parent directory for importing model_loader directly ──
# This allows the dashboard to work BOTH with the API (via requests)
# and as a standalone tool (direct model loading) for demo purposes.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "phase3_api"))

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────
API_URL = os.environ.get("API_URL", "http://localhost:8000")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "phase2_modeling", "artifacts")
SCORED_DATA_PATH = os.path.join(ARTIFACTS_DIR, "scored_customers.csv")


# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG — Must be the first Streamlit command
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retention Safeguard System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────
# CUSTOM STYLING
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* KPI metric cards */
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #e0e3e8;
        border-radius: 8px;
        padding: 12px 16px;
    }
    /* Quadrant color coding */
    .persuadable { color: #28a745; font-weight: bold; }
    .sleeping-dog { color: #dc3545; font-weight: bold; }
    .sure-thing { color: #17a2b8; font-weight: bold; }
    .lost-cause { color: #6c757d; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_scored_data() -> pd.DataFrame:
    """
    Load pre-scored customer data from Phase 2 artifacts.

    WHY CACHE?
        @st.cache_data ensures the CSV is loaded ONCE and reused
        across all user interactions. Without caching, every slider
        movement would re-read the file from disk.
    """
    if not os.path.exists(SCORED_DATA_PATH):
        st.error(
            f"Scored data not found at: {SCORED_DATA_PATH}\n\n"
            f"Please run Phase 2 scripts first."
        )
        st.stop()

    df = pd.read_csv(SCORED_DATA_PATH)
    return df


def check_api_health() -> dict:
    """Check if the FastAPI backend is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def predict_via_api(customer_data: dict) -> dict:
    """Send a prediction request to the FastAPI backend."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=customer_data,
            timeout=5,
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} — {response.text}")
            return None
    except requests.ConnectionError:
        st.warning("API not running. Using pre-scored data only.")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────
def render_sidebar(df: pd.DataFrame) -> dict:
    """
    Sidebar controls for the marketing manager.

    DESIGN DECISION:
        The sidebar contains BUSINESS controls, not technical parameters.
        The marketing manager sets their budget and cost-per-contact.
        The dashboard handles the optimization math behind the scenes.
    """
    st.sidebar.title("🛡️ Retention Safeguard")
    st.sidebar.markdown("---")

    # ── API Status ──
    api_health = check_api_health()
    if api_health:
        st.sidebar.success(f"✅ API Online — {api_health['model_type']}")
    else:
        st.sidebar.warning("⚠️ API Offline — Using pre-scored data")

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Campaign Settings")

    # ── Marketing Budget ──
    # This is the KEY business input. Everything else flows from this.
    max_revenue = float(df[df["quadrant"] == "Persuadable"]["value_at_risk"].sum())
    budget = st.sidebar.slider(
        "Marketing Budget ($)",
        min_value=1000,
        max_value=100000,
        value=25000,
        step=1000,
        help="Total budget available for retention campaign"
    )

    # ── Cost per Contact ──
    cost_per_contact = st.sidebar.slider(
        "Cost per Contact ($)",
        min_value=5,
        max_value=100,
        value=25,
        step=5,
        help="Average cost to reach one customer (email, call, offer value)"
    )

    # ── Minimum Uplift Threshold ──
    min_uplift = st.sidebar.slider(
        "Minimum Uplift Score",
        min_value=0.0,
        max_value=0.5,
        value=0.01,
        step=0.01,
        help="Only target customers with uplift above this threshold"
    )

    # ── Filters ──
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Filters")

    contract_filter = st.sidebar.multiselect(
        "Contract Type",
        options=df["contract_type"].unique().tolist(),
        default=df["contract_type"].unique().tolist(),
    )

    internet_filter = st.sidebar.multiselect(
        "Internet Service",
        options=df["internet_service"].unique().tolist(),
        default=df["internet_service"].unique().tolist(),
    )

    return {
        "budget": budget,
        "cost_per_contact": cost_per_contact,
        "min_uplift": min_uplift,
        "contract_filter": contract_filter,
        "internet_filter": internet_filter,
        "api_health": api_health,
    }


# ─────────────────────────────────────────────────────────────────────
# KPI METRICS ROW
# ─────────────────────────────────────────────────────────────────────
def render_kpi_metrics(df: pd.DataFrame, settings: dict) -> None:
    """
    Top-level KPI metrics that a marketing manager cares about.

    DESIGN RULE: Lead with dollars, not probabilities.
    A CMO doesn't think in AUC or uplift scores.
    They think in: "How much money is at risk? How much can I save?"
    """
    st.markdown("## 📈 Executive Summary")

    persuadables = df[df["quadrant"] == "Persuadable"]
    sleeping_dogs = df[df["quadrant"] == "Sleeping Dog"]
    total_revenue_at_risk = df[df["churn_binary"] == 1]["revenue_at_risk"].sum()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="💰 Total Revenue at Risk",
            value=f"${total_revenue_at_risk:,.0f}",
            help="Annual revenue from customers predicted to churn"
        )

    with col2:
        st.metric(
            label="🎯 Persuadable Customers",
            value=f"{len(persuadables):,}",
            delta=f"${persuadables['value_at_risk'].sum():,.0f} saveable",
            delta_color="normal",
            help="Customers whose churn can be reduced with an offer"
        )

    with col3:
        st.metric(
            label="🛑 Sleeping Dogs",
            value=f"{len(sleeping_dogs):,}",
            delta=f"-${abs(sleeping_dogs['value_at_risk'].sum()):,.0f} at risk",
            delta_color="inverse",
            help="Customers who would CHURN if contacted — do NOT target"
        )

    with col4:
        avg_uplift = persuadables["uplift_score"].mean() if len(persuadables) > 0 else 0
        st.metric(
            label="📊 Avg Persuadable Uplift",
            value=f"{avg_uplift:.1%}",
            help="Average churn reduction for Persuadable segment"
        )


# ─────────────────────────────────────────────────────────────────────
# BUDGET OPTIMIZER
# ─────────────────────────────────────────────────────────────────────
def render_budget_optimizer(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """
    The core business tool: given a budget, who should we target?

    OPTIMIZATION LOGIC:
        1. Filter to Persuadable customers with uplift > threshold
        2. Sort by value_at_risk (highest ROI first)
        3. Select as many as the budget allows
        4. Show projected ROI

    THIS IS THE KEY DIFFERENTIATOR of the project.
    Basic churn models say "these customers might leave."
    This tool says "spend $25K targeting THESE 400 customers
    to save $180K in revenue — that's a 7.2x ROI."
    """
    st.markdown("## 💼 Budget Optimizer")
    st.markdown(
        "Allocate your marketing budget to the highest-ROI Persuadable targets. "
        "Customers are ranked by value-at-risk (saveable revenue)."
    )

    budget = settings["budget"]
    cost_per_contact = settings["cost_per_contact"]
    min_uplift = settings["min_uplift"]

    # ── Filter to targetable Persuadables ──
    targets = df[
        (df["quadrant"] == "Persuadable")
        & (df["uplift_score"] >= min_uplift)
        & (df["contract_type"].isin(settings["contract_filter"]))
        & (df["internet_service"].isin(settings["internet_filter"]))
    ].copy()

    # ── Sort by value at risk (highest first) ──
    targets = targets.sort_values("value_at_risk", ascending=False)

    # ── How many can we afford? ──
    max_contacts = int(budget / cost_per_contact)
    selected = targets.head(max_contacts)

    # ── ROI Calculation ──
    campaign_cost = len(selected) * cost_per_contact
    projected_savings = selected["value_at_risk"].sum()
    roi = (projected_savings / campaign_cost - 1) * 100 if campaign_cost > 0 else 0

    # ── Display ROI metrics ──
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🎯 Customers Targeted", f"{len(selected):,}")
    with col2:
        st.metric("💸 Campaign Cost", f"${campaign_cost:,.0f}")
    with col3:
        st.metric("💰 Projected Savings", f"${projected_savings:,.0f}")
    with col4:
        roi_color = "normal" if roi > 0 else "inverse"
        st.metric("📈 Projected ROI", f"{roi:.1f}%")

    # ── Efficiency message ──
    if roi > 100:
        st.success(
            f"🎉 **Excellent ROI!** For every $1 spent, you save "
            f"${projected_savings/campaign_cost:.2f} in revenue. "
            f"Targeting {len(selected)} of {len(targets)} eligible Persuadables."
        )
    elif roi > 0:
        st.info(
            f"✅ **Positive ROI.** Campaign is profitable. Consider increasing "
            f"budget to capture more Persuadables ({len(targets) - len(selected)} remaining)."
        )
    else:
        st.warning(
            f"⚠️ **Negative ROI.** Cost per contact may be too high, or "
            f"remaining Persuadables have low value. Consider reducing cost per contact."
        )

    return selected


# ─────────────────────────────────────────────────────────────────────
# QUADRANT DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────
def render_quadrant_analysis(df: pd.DataFrame) -> None:
    """
    Visual breakdown of the four uplift quadrants.

    Uses Streamlit's native bar chart for simplicity.
    In a production dashboard, you'd use Plotly for interactivity.
    """
    st.markdown("## 🔄 Customer Segmentation — Uplift Quadrants")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Quadrant Distribution")
        quadrant_counts = df["quadrant"].value_counts()

        # Color-coded display
        for quadrant in ["Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog"]:
            count = quadrant_counts.get(quadrant, 0)
            pct = count / len(df) * 100

            emoji_map = {
                "Persuadable": "🎯",
                "Sure Thing": "✅",
                "Lost Cause": "❌",
                "Sleeping Dog": "🛑",
            }
            emoji = emoji_map.get(quadrant, "")

            value_total = df[df["quadrant"] == quadrant]["value_at_risk"].sum()

            st.markdown(
                f"**{emoji} {quadrant}:** {count:,} customers ({pct:.1f}%) — "
                f"Value: ${value_total:,.0f}"
            )

    with col2:
        st.markdown("### Quadrant by Revenue Impact")
        quadrant_revenue = df.groupby("quadrant")["value_at_risk"].sum().reset_index()
        quadrant_revenue.columns = ["Quadrant", "Total Value at Risk ($)"]
        quadrant_revenue = quadrant_revenue.sort_values("Total Value at Risk ($)", ascending=False)
        st.bar_chart(
            quadrant_revenue.set_index("Quadrant"),
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────
# SEGMENT DEEP DIVE
# ─────────────────────────────────────────────────────────────────────
def render_segment_analysis(df: pd.DataFrame) -> None:
    """Breakdown by contract type and internet service."""
    st.markdown("## 📊 Segment Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Churn Rate by Contract Type")
        contract_stats = df.groupby("contract_type").agg(
            customers=("customer_id", "count"),
            churn_rate=("churn_binary", "mean"),
            avg_uplift=("uplift_score", "mean"),
            total_value=("value_at_risk", "sum"),
        ).round(4)
        contract_stats["churn_rate"] = (contract_stats["churn_rate"] * 100).round(1)
        contract_stats["avg_uplift"] = (contract_stats["avg_uplift"] * 100).round(2)
        contract_stats["total_value"] = contract_stats["total_value"].round(0)
        contract_stats.columns = ["Customers", "Churn Rate %", "Avg Uplift %", "Value at Risk $"]
        st.dataframe(contract_stats, use_container_width=True)

    with col2:
        st.markdown("### Churn Rate by Internet Service")
        internet_stats = df.groupby("internet_service").agg(
            customers=("customer_id", "count"),
            churn_rate=("churn_binary", "mean"),
            avg_uplift=("uplift_score", "mean"),
            total_value=("value_at_risk", "sum"),
        ).round(4)
        internet_stats["churn_rate"] = (internet_stats["churn_rate"] * 100).round(1)
        internet_stats["avg_uplift"] = (internet_stats["avg_uplift"] * 100).round(2)
        internet_stats["total_value"] = internet_stats["total_value"].round(0)
        internet_stats.columns = ["Customers", "Churn Rate %", "Avg Uplift %", "Value at Risk $"]
        st.dataframe(internet_stats, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# TARGET LIST & DOWNLOAD
# ─────────────────────────────────────────────────────────────────────
def render_target_list(selected: pd.DataFrame) -> None:
    """
    Show the targeted customer list and provide CSV download.

    BUSINESS WORKFLOW:
        Marketing manager adjusts budget slider → sees the optimal target list
        → downloads as CSV → uploads to their email/CRM tool → launches campaign.

    This is where the model output becomes a BUSINESS ACTION.
    """
    st.markdown("## 📋 Target List — Campaign-Ready")

    if len(selected) == 0:
        st.warning("No customers selected. Adjust budget or filters.")
        return

    # ── Display columns relevant to marketing (not model internals) ──
    display_cols = [
        "customer_id", "contract_type", "internet_service",
        "tenure_months", "monthly_charges", "clv",
        "uplift_score", "quadrant", "value_at_risk",
    ]

    # Only show columns that exist
    available_cols = [c for c in display_cols if c in selected.columns]
    display_df = selected[available_cols].copy()

    # Format for readability
    if "uplift_score" in display_df.columns:
        display_df["uplift_score"] = display_df["uplift_score"].apply(lambda x: f"{x:.1%}")
    if "clv" in display_df.columns:
        display_df["clv"] = display_df["clv"].apply(lambda x: f"${x:,.0f}")
    if "value_at_risk" in display_df.columns:
        display_df["value_at_risk"] = display_df["value_at_risk"].apply(lambda x: f"${x:,.0f}")
    if "monthly_charges" in display_df.columns:
        display_df["monthly_charges"] = display_df["monthly_charges"].apply(lambda x: f"${x:.2f}")

    st.dataframe(display_df.head(50), use_container_width=True)

    if len(selected) > 50:
        st.caption(f"Showing top 50 of {len(selected):,} targeted customers.")

    # ── CSV Download ──
    csv_data = selected[available_cols].to_csv(index=False)
    st.download_button(
        label="📥 Download Target List (CSV)",
        data=csv_data,
        file_name="retention_campaign_targets.csv",
        mime="text/csv",
        help="Download the full targeted customer list for campaign execution"
    )


# ─────────────────────────────────────────────────────────────────────
# INDIVIDUAL CUSTOMER SCORER (connects to API)
# ─────────────────────────────────────────────────────────────────────
def render_individual_scorer(api_health: dict) -> None:
    """
    Score a single customer via the FastAPI backend.

    This demonstrates the full-stack integration:
    Streamlit (frontend) → FastAPI (backend) → T-Learner (model)
    """
    st.markdown("## 🔍 Score Individual Customer")

    if not api_health:
        st.info(
            "💡 Start the API to enable real-time scoring: "
            "`cd phase3_api && uvicorn main:app --port 8000`"
        )

    with st.expander("Enter Customer Details", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", [
                "Month-to-month", "One year", "Two year"
            ])

        with col2:
            internet = st.selectbox("Internet Service", [
                "Fiber optic", "DSL", "No"
            ])
            security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])

        with col3:
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0)
            total = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0)
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

        if st.button("🔮 Score Customer", type="primary"):
            customer_data = {
                "gender": gender,
                "senior_citizen": senior,
                "partner": partner,
                "dependents": dependents,
                "tenure_months": tenure,
                "contract_type": contract,
                "internet_service": internet,
                "online_security": security,
                "online_backup": backup,
                "device_protection": protection,
                "tech_support": support,
                "streaming_tv": streaming_tv,
                "streaming_movies": streaming_movies,
                "phone_service": phone,
                "multiple_lines": multiple_lines,
                "monthly_charges": monthly,
                "total_charges": total,
                "payment_method": payment,
                "paperless_billing": paperless,
            }

            if api_health:
                result = predict_via_api(customer_data)
                if result:
                    _display_prediction_result(result)
            else:
                # Fallback: use model_loader directly
                try:
                    from model_loader import ModelLoader
                    loader = ModelLoader(model_dir=ARTIFACTS_DIR)
                    loader.load()
                    result = loader.predict(customer_data)
                    _display_prediction_result(result)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")


def _display_prediction_result(result: dict) -> None:
    """Display a single prediction result with visual formatting."""
    quadrant = result.get("quadrant", "Unknown")

    # Color coding by quadrant
    color_map = {
        "Persuadable": "🟢",
        "Sure Thing": "🔵",
        "Lost Cause": "⚪",
        "Sleeping Dog": "🔴",
    }
    icon = color_map.get(quadrant, "⚪")

    st.markdown(f"### {icon} Result: **{quadrant}**")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Churn Risk", f"{result['churn_risk']:.1%}")
    with col2:
        st.metric("Churn w/ Offer", f"{result['churn_risk_with_offer']:.1%}")
    with col3:
        st.metric("Uplift Score", f"{result['uplift_score']:+.1%}")
    with col4:
        st.metric("Value at Risk", f"${result['value_at_risk']:,.0f}")

    # Recommendation box
    st.info(f"**Recommendation:** {result['recommendation']}")
    st.caption(f"Confidence: {result['confidence']} | CLV: ${result['customer_lifetime_value']:,.0f}")


# ─────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────
def main():
    """
    Main application entry point.

    LAYOUT:
        1. Header with project title
        2. KPI metrics row (revenue at risk, persuadable count)
        3. Budget optimizer (slider → ROI calculation)
        4. Quadrant analysis (visual breakdown)
        5. Segment analysis (contract × internet)
        6. Target list with CSV download
        7. Individual customer scorer (API integration)
    """
    # ── Header ──
    st.title("🛡️ Proactive Retention & Revenue Safeguard System")
    st.markdown(
        "*Causal AI-powered customer retention — identifying who you can save, "
        "not just who will leave.*"
    )
    st.markdown("---")

    # ── Load data ──
    df = load_scored_data()

    # ── Sidebar settings ──
    settings = render_sidebar(df)

    # ── Apply filters ──
    df_filtered = df[
        (df["contract_type"].isin(settings["contract_filter"]))
        & (df["internet_service"].isin(settings["internet_filter"]))
    ]

    # ── KPI Metrics ──
    render_kpi_metrics(df_filtered, settings)
    st.markdown("---")

    # ── Budget Optimizer ──
    selected_targets = render_budget_optimizer(df_filtered, settings)
    st.markdown("---")

    # ── Quadrant Analysis ──
    render_quadrant_analysis(df_filtered)
    st.markdown("---")

    # ── Segment Analysis ──
    render_segment_analysis(df_filtered)
    st.markdown("---")

    # ── Target List ──
    render_target_list(selected_targets)
    st.markdown("---")

    # ── Individual Scorer ──
    render_individual_scorer(settings["api_health"])

    # ── Footer ──
    st.markdown("---")
    st.markdown(
        "*Built by [Utsav Khadka](https://github.com/UtsavKhadka-Analyst) "
        "| MS Analytics Capstone Project | Powered by T-Learner Uplift Modeling*"
    )


if __name__ == "__main__":
    main()