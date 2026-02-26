"""
=============================================================================
PHASE 1 | Script 1 of 3: fetch_and_enrich_data.py
=============================================================================
Proactive Retention & Revenue Safeguard System
Author: Utsav Khadka (github.com/UtsavKhadka-Analyst)

PURPOSE:
    This script does two things:
    1. Downloads the IBM Telco Customer Churn dataset (7,043 customers)
    2. Enriches it with THREE synthetic columns that enable causal analysis

WHY THESE THREE COLUMNS?
    The original dataset has a "Churn" label but NO treatment variable.
    Without a treatment variable, we can only PREDICT churn — we cannot
    estimate WHO CAN BE SAVED by an intervention. That's the gap between
    a basic ML project and a causal AI project.

    We add:
    - Marketing_Offer_Given (0/1): The TREATMENT variable
    - Customer_Lifetime_Value ($): Converts churn probability → dollar impact
    - Propensity_Score: P(Treatment=1 | X) — needed for bias correction

IMPORTANT DESIGN DECISION — NON-RANDOM TREATMENT ASSIGNMENT:
    In real companies, marketing teams don't randomly assign offers.
    They target customers they THINK are at risk. This creates SELECTION BIAS:
    the treatment group systematically differs from the control group.

    We simulate this intentionally so that Phase 2's causal methods
    (IPW, T-Learner, X-Learner) have a real problem to solve.
    If treatment were random, you wouldn't need causal inference at all.

INTERVIEW PREP:
    Q: "Why didn't you assign treatment randomly?"
    A: "Because random assignment would make causal correction unnecessary.
       I intentionally introduced selection bias to demonstrate that I
       understand WHY methods like IPW exist, not just how to code them."
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import os
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────
# We use a fixed random seed so that EVERY run produces identical results.
# Reproducibility is non-negotiable in data science — if your reviewer
# can't reproduce your numbers, they can't trust your work.
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Treatment rate: ~30% of customers received a marketing offer.
# This is realistic — most companies can't afford to treat everyone,
# and they shouldn't (Sleeping Dogs would be harmed by contact).
TREATMENT_RATE = 0.30

# Output path
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "telco_churn_enriched.csv")


def load_telco_dataset() -> pd.DataFrame:
    """
    Load the IBM Telco Customer Churn dataset.

    ABOUT THE DATASET:
        - 7,043 customers from a fictional telco company
        - 21 features: demographics, account info, service subscriptions
        - Binary target: "Churn" (Yes/No)
        - Publicly available — no licensing issues for portfolio projects

    WHY TELCO?
        Churn patterns in telecom generalize to SaaS, banking, insurance,
        and streaming. When a hiring manager sees this, they map it to
        their own business instantly. That's the goal.
    """
    print("[INFO] Loading IBM Telco Customer Churn dataset...")

    # ── Try to load from Kaggle's common URL ──
    # The dataset is hosted on multiple public sources.
    # We try the most reliable URL first.
    urls = [
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
        "https://raw.githubusercontent.com/dsrscientist/dataset1/master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    ]

    df = None
    for url in urls:
        try:
            df = pd.read_csv(url)
            print(f"[SUCCESS] Loaded {len(df)} rows from remote source.")
            break
        except Exception as e:
            print(f"[WARNING] Could not load from {url}: {e}")
            continue

    # ── Fallback: generate a representative synthetic dataset ──
    # If network is unavailable (common in corporate environments),
    # we generate data with the same schema and distributions.
    if df is None:
        print("[FALLBACK] Generating synthetic Telco dataset locally...")
        df = _generate_fallback_dataset()
        print(f"[SUCCESS] Generated {len(df)} synthetic rows.")

    return df


def _generate_fallback_dataset(n: int = 7043) -> pd.DataFrame:
    """
    Generate a synthetic dataset matching the IBM Telco schema.

    This fallback ensures the pipeline runs even without internet.
    The distributions are calibrated to match the real dataset:
    - ~26.5% churn rate
    - Tenure: 1-72 months
    - MonthlyCharges: $18-$118
    - Mix of contract types, payment methods, services
    """
    tenure = np.random.exponential(scale=32, size=n).clip(1, 72).astype(int)

    # ── Contract type heavily influences churn ──
    # Month-to-month customers churn at ~42%, two-year at ~3%
    # This is the single strongest predictor in the real dataset.
    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n,
        p=[0.55, 0.24, 0.21]  # Real dataset proportions
    )

    # ── Monthly charges correlate with services subscribed ──
    base_charge = np.random.normal(65, 25, n).clip(18, 118)

    # ── Internet service type affects both charges and churn ──
    internet = np.random.choice(
        ["DSL", "Fiber optic", "No"],
        size=n,
        p=[0.34, 0.44, 0.22]
    )

    # ── Churn probability: driven by contract, tenure, charges ──
    # This is a SIMPLIFIED version of the real relationship.
    # The actual dataset has more nuance, but this captures the key drivers.
    churn_prob = np.zeros(n)
    churn_prob += np.where(np.array(contract) == "Month-to-month", 0.30, 0.0)
    churn_prob += np.where(np.array(contract) == "One year", 0.08, 0.0)
    churn_prob += np.where(np.array(contract) == "Two year", 0.02, 0.0)
    churn_prob += np.where(tenure < 12, 0.15, 0.0)
    churn_prob += np.where(base_charge > 80, 0.05, 0.0)
    churn_prob += np.where(np.array(internet) == "Fiber optic", 0.05, 0.0)
    churn_prob = churn_prob.clip(0.02, 0.85)

    churn = np.random.binomial(1, churn_prob).astype(str)
    churn = np.where(churn == "1", "Yes", "No")

    # ── Remaining features ──
    gender = np.random.choice(["Male", "Female"], n)
    senior = np.random.binomial(1, 0.16, n)
    partner = np.random.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = np.random.choice(["Yes", "No"], n, p=[0.30, 0.70])
    phone = np.random.choice(["Yes", "No"], n, p=[0.90, 0.10])
    multiple_lines = np.random.choice(["Yes", "No", "No phone service"], n, p=[0.42, 0.48, 0.10])
    security = np.random.choice(["Yes", "No", "No internet service"], n, p=[0.29, 0.49, 0.22])
    backup = np.random.choice(["Yes", "No", "No internet service"], n, p=[0.34, 0.44, 0.22])
    protection = np.random.choice(["Yes", "No", "No internet service"], n, p=[0.34, 0.44, 0.22])
    support = np.random.choice(["Yes", "No", "No internet service"], n, p=[0.29, 0.49, 0.22])
    streaming_tv = np.random.choice(["Yes", "No", "No internet service"], n, p=[0.38, 0.40, 0.22])
    streaming_movies = np.random.choice(["Yes", "No", "No internet service"], n, p=[0.39, 0.39, 0.22])
    paperless = np.random.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n, p=[0.34, 0.23, 0.22, 0.21]
    )
    total_charges = (tenure * base_charge * np.random.uniform(0.9, 1.1, n)).round(2)

    df = pd.DataFrame({
        "customerID": [f"CUST-{i:05d}" for i in range(n)],
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": base_charge.round(2),
        "TotalCharges": total_charges,
        "Churn": churn,
    })

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset.

    BUSINESS CONTEXT:
        Real-world data is ALWAYS messy. The Telco dataset has:
        - TotalCharges stored as string (has whitespace for new customers)
        - Missing values for customers with tenure = 0

    WHAT YOU LEARN HERE:
        Data cleaning is 60-80% of a data scientist's actual job.
        This function handles the unglamorous but critical work.
    """
    print("[INFO] Cleaning dataset...")

    # ── Fix TotalCharges: stored as string, has whitespace ──
    # New customers (tenure=0) have " " instead of 0.0
    # This is a REAL issue in the actual IBM dataset.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # ── Convert Churn to binary (1 = churned, 0 = stayed) ──
    # ML models need numeric targets. We keep the original for reference.
    df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)

    # ── Verify no nulls remain ──
    null_count = df.isnull().sum().sum()
    print(f"[INFO] Remaining null values: {null_count}")

    print(f"[INFO] Churn rate: {df['Churn_Binary'].mean():.1%}")
    print(f"[INFO] Dataset shape: {df.shape}")

    return df


def compute_propensity_score(df: pd.DataFrame) -> np.ndarray:
    """
    Estimate the propensity score: P(Treatment=1 | X)

    CAUSAL MATH — WHY THIS MATTERS:
        The propensity score e(x) = P(T=1 | X=x) answers:
        "Given this customer's features, how likely were they to
        receive a marketing offer?"

        In real companies, treatment assignment is NOT random.
        Marketing teams target customers they believe are at risk.
        This means the treatment group is systematically different
        from the control group — that's SELECTION BIAS.

        The propensity score lets us correct for this bias in Phase 2
        using Inverse Probability Weighting (IPW):
        - Treated units weighted by: 1 / e(x)
        - Control units weighted by: 1 / (1 - e(x))
        This creates a "pseudo-population" where treatment is
        effectively random, allowing valid causal estimates.

    IMPLEMENTATION:
        We use Logistic Regression to estimate propensity scores.
        In production, you might use GBM or a more flexible model,
        but logistic regression is standard and interpretable.

    INTERVIEW PREP:
        Q: "What assumptions does propensity score adjustment require?"
        A: "Positivity (every customer has some chance of being treated),
           unconfoundedness (no hidden confounders after conditioning on X),
           and correct specification of the propensity model."
    """
    print("[INFO] Computing propensity scores via Logistic Regression...")

    # ── Encode categorical features for the propensity model ──
    # We need numeric inputs for logistic regression.
    feature_cols = [
        "tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
        "Contract", "InternetService", "PaymentMethod",
        "Partner", "Dependents", "PaperlessBilling"
    ]

    df_encoded = df[feature_cols].copy()

    # Label-encode categorical columns
    label_encoders = {}
    for col in df_encoded.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    # ── Fit logistic regression ──
    # The propensity score model predicts treatment assignment from features.
    # Higher risk customers → higher propensity → more likely to get offers.
    #
    # We create a "risk signal" that drives treatment assignment:
    # tenure (shorter = riskier), MonthlyCharges (higher = more to lose),
    # Contract type (month-to-month = riskiest)
    risk_signal = (
        (df["tenure"] < 24).astype(float) * 0.3
        + (df["MonthlyCharges"] > 70).astype(float) * 0.2
        + (df["Contract"] == "Month-to-month").astype(float) * 0.3
        + (df["Churn_Binary"]).astype(float) * 0.15
        + np.random.normal(0, 0.1, len(df))  # Noise for realism
    ).clip(0.05, 0.95)

    # ── Fit a proper logistic regression on features ──
    # This gives us smooth, well-calibrated propensity scores.
    propensity_model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)

    # Create binary "treatment" based on risk signal for fitting
    treatment_for_fit = (risk_signal > np.percentile(risk_signal, 100 * (1 - TREATMENT_RATE))).astype(int)

    propensity_model.fit(df_encoded, treatment_for_fit)
    propensity_scores = propensity_model.predict_proba(df_encoded)[:, 1]

    print(f"[INFO] Propensity score range: [{propensity_scores.min():.3f}, {propensity_scores.max():.3f}]")
    print(f"[INFO] Propensity score mean:  {propensity_scores.mean():.3f}")

    return propensity_scores


def assign_treatment(df: pd.DataFrame, propensity_scores: np.ndarray) -> pd.Series:
    """
    Assign treatment (Marketing_Offer_Given) based on propensity scores.

    CRITICAL DESIGN DECISION — NON-RANDOM ASSIGNMENT:
        We use propensity scores to BIAS treatment assignment.
        Customers with higher propensity scores are MORE LIKELY
        to receive an offer. This simulates real-world behavior:
        marketing teams target who they think needs help.

        This is the OPPOSITE of a randomized experiment.
        It creates selection bias that Phase 2 must correct.

    WHY THIS MATTERS FOR YOUR LEARNING:
        If treatment were random, a simple difference-in-means
        between treatment and control groups would give you the
        Average Treatment Effect (ATE). No fancy methods needed.

        But with biased assignment, the naive comparison is WRONG
        because the groups aren't comparable. You NEED causal methods
        (T-Learner, X-Learner, IPW) to get valid estimates.

        This is the entire reason this project exists.
    """
    print("[INFO] Assigning non-random treatment based on propensity scores...")

    # ── Treatment probability proportional to propensity score ──
    # Higher propensity score → higher chance of receiving offer
    treatment = np.random.binomial(1, propensity_scores)

    # ── Ensure we hit approximately the target treatment rate ──
    actual_rate = treatment.mean()
    print(f"[INFO] Treatment rate: {actual_rate:.1%} (target: {TREATMENT_RATE:.0%})")

    return pd.Series(treatment, name="Marketing_Offer_Given")


def compute_clv(df: pd.DataFrame) -> pd.Series:
    """
    Compute Customer Lifetime Value (CLV).

    BUSINESS CONTEXT:
        Churn probability alone doesn't tell you WHO MATTERS MOST.
        A customer with 80% churn probability but $20/month revenue
        is less important than a customer with 40% churn probability
        and $100/month revenue.

        CLV converts "probability of leaving" into "dollars at risk."
        This is what a CMO or VP of Marketing actually cares about.

    FORMULA:
        CLV = Monthly Revenue × Expected Remaining Lifetime

        Expected Remaining Lifetime = tenure_factor × contract_multiplier
        - Longer-tenured customers have proven loyalty → higher multiplier
        - Longer contracts → more guaranteed future revenue

    WHY THIS IS SIMPLIFIED:
        A real CLV model would include:
        - Discount rate (time value of money)
        - Margin, not revenue
        - Probability of renewal at contract end
        - Cross-sell/upsell potential

        For this portfolio project, a simplified CLV is sufficient
        to demonstrate the ROI calculation. Document the simplification.
    """
    print("[INFO] Computing Customer Lifetime Value (CLV)...")

    # ── Tenure factor: longer tenure → more expected future months ──
    # Customers who've stayed 5 years are likely to stay longer
    # than customers who joined last month. This is empirically validated.
    tenure_factor = np.log1p(df["tenure"]) * 3  # Log scale dampens extreme values

    # ── Contract multiplier: longer contracts → more guaranteed revenue ──
    contract_map = {
        "Month-to-month": 6,   # ~6 months expected remaining
        "One year": 14,        # ~14 months (remaining contract + renewal probability)
        "Two year": 26,        # ~26 months
    }
    contract_multiplier = df["Contract"].map(contract_map)

    # ── CLV = MonthlyCharges × (tenure_factor + contract_multiplier) ──
    clv = (df["MonthlyCharges"] * (tenure_factor + contract_multiplier)).round(2)

    # ── Add realistic noise (±15%) ──
    noise = np.random.uniform(0.85, 1.15, len(df))
    clv = (clv * noise).round(2)

    print(f"[INFO] CLV range: [${clv.min():,.2f}, ${clv.max():,.2f}]")
    print(f"[INFO] CLV mean:  ${clv.mean():,.2f}")

    return pd.Series(clv, name="Customer_Lifetime_Value")


def enrich_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrate the full enrichment pipeline.

    This function ties together:
    1. Propensity score computation
    2. Treatment assignment (with selection bias)
    3. CLV calculation

    The output is a single DataFrame ready for database loading (Script 2)
    and SQL analysis (Script 3).
    """
    print("\n" + "=" * 60)
    print("ENRICHMENT PIPELINE")
    print("=" * 60)

    # ── Step 1: Propensity Scores ──
    propensity_scores = compute_propensity_score(df)
    df["Propensity_Score"] = propensity_scores.round(4)

    # ── Step 2: Treatment Assignment ──
    df["Marketing_Offer_Given"] = assign_treatment(df, propensity_scores)

    # ── Step 3: Customer Lifetime Value ──
    df["Customer_Lifetime_Value"] = compute_clv(df)

    # ── Sanity Checks ──
    # These assertions catch data issues before they propagate downstream.
    # In production, these would be data quality tests (e.g., Great Expectations).
    assert df["Marketing_Offer_Given"].isin([0, 1]).all(), "Treatment must be binary"
    assert df["Customer_Lifetime_Value"].min() > 0, "CLV must be positive"
    assert df["Propensity_Score"].between(0, 1).all(), "Propensity scores must be in [0,1]"
    assert df["Churn_Binary"].isin([0, 1]).all(), "Churn must be binary"

    print("\n[SUCCESS] All sanity checks passed.")
    return df


def print_summary_statistics(df: pd.DataFrame) -> None:
    """
    Print a summary that demonstrates understanding of the data.

    WHAT AN INTERVIEWER LOOKS FOR:
        Can you describe your data clearly and spot issues?
        This function shows you've inspected the data, not just fed it
        into a model blindly.
    """
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total customers:     {len(df):,}")
    print(f"Features:            {df.shape[1]}")
    print(f"Churn rate:          {df['Churn_Binary'].mean():.1%}")
    print(f"Treatment rate:      {df['Marketing_Offer_Given'].mean():.1%}")
    print(f"Avg CLV:             ${df['Customer_Lifetime_Value'].mean():,.2f}")
    print(f"Avg propensity:      {df['Propensity_Score'].mean():.3f}")

    # ── Key insight: show selection bias ──
    # Treatment group should have HIGHER churn rate than control
    # because we targeted high-risk customers. This is the bias
    # that Phase 2 must correct.
    treated = df[df["Marketing_Offer_Given"] == 1]
    control = df[df["Marketing_Offer_Given"] == 0]

    print(f"\n--- Selection Bias Check ---")
    print(f"Churn rate (Treatment): {treated['Churn_Binary'].mean():.1%}")
    print(f"Churn rate (Control):   {control['Churn_Binary'].mean():.1%}")
    print(f"Naive difference:       {treated['Churn_Binary'].mean() - control['Churn_Binary'].mean():.1%}")
    print(f"\n[NOTE] The treatment group has HIGHER churn — this is EXPECTED.")
    print(f"       It's selection bias, not a treatment effect.")
    print(f"       Naive comparison would wrongly conclude the offer INCREASES churn.")
    print(f"       Phase 2 corrects this with causal methods.")


# ─────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────
def main():
    """
    Execute the full Phase 1 data pipeline.

    Pipeline: Load → Clean → Enrich → Validate → Save
    """
    print("=" * 60)
    print("PHASE 1: HYBRID DATASET & ENRICHMENT PIPELINE")
    print("Proactive Retention & Revenue Safeguard System")
    print("=" * 60 + "\n")

    # Step 1: Load raw data
    df = load_telco_dataset()

    # Step 2: Clean
    df = clean_dataset(df)

    # Step 3: Enrich with causal columns
    df = enrich_dataset(df)

    # Step 4: Summary
    print_summary_statistics(df)

    # Step 5: Save
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[SAVED] Enriched dataset → {OUTPUT_PATH}")
    print(f"[INFO]  {len(df)} rows × {df.shape[1]} columns")
    print(f"\n{'=' * 60}")
    print("Phase 1, Script 1 COMPLETE. Proceed to Script 2 (load_to_db.py)")
    print(f"{'=' * 60}")

    return df


if __name__ == "__main__":
    main()