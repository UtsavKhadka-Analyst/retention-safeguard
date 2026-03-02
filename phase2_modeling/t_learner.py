"""
=============================================================================
PHASE 2 | Script 1 of 5: t_learner.py
=============================================================================
Proactive Retention & Revenue Safeguard System
Author: Utsav Khadka (github.com/UtsavKhadka-Analyst)

PURPOSE:
    Implement the T-Learner (Two-Model Learner) for uplift estimation.
    This is the BASELINE causal method. Phase 2 also includes X-Learner
    for comparison.

WHAT IS A T-LEARNER?
    Instead of one model, we train TWO:
      Model 0 (Control):   trained ONLY on customers who did NOT get an offer
      Model 1 (Treatment): trained ONLY on customers who DID get an offer

    For any new customer with features X:
      μ₁(X) = Model 1's prediction = P(churn | X, treated)
      μ₀(X) = Model 0's prediction = P(churn | X, not treated)

      Uplift = τ(X) = μ₀(X) - μ₁(X)

    NOTE: We compute μ₀ - μ₁ (not μ₁ - μ₀) because:
      - Y = 1 means churn (bad outcome)
      - Positive uplift = the offer REDUCES churn = GOOD
      - If μ₀ = 0.60 and μ₁ = 0.30, uplift = 0.30 → the offer
        reduces churn probability by 30 percentage points

THE FOUR QUADRANTS:
    Based on the uplift score τ(X), every customer falls into one of:

    1. PERSUADABLES (τ >> 0): Would churn without offer, stay with offer
       → These are your GOLD. Target them. Highest ROI.

    2. SURE THINGS (τ ≈ 0, low churn both ways): Stay regardless
       → Don't waste budget. They're already loyal.

    3. LOST CAUSES (τ ≈ 0, high churn both ways): Leave regardless
       → Don't waste budget. Nothing will save them.

    4. SLEEPING DOGS (τ << 0): Would stay WITHOUT offer, churn WITH offer
       → NEVER contact them. The offer annoys them and causes churn.


=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
import pickle
import os
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PHASE1_DIR = os.path.join(PROJECT_ROOT, "phase1_data")
INPUT_PATH = os.path.join(PHASE1_DIR, "modeling_dataset.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "artifacts")


def load_and_prepare_data(path: str) -> tuple:
    """
    Load the modeling dataset from Phase 1 and prepare for T-Learner.

    FEATURE ENGINEERING DECISIONS:
        - Label-encode categoricals (XGBoost/GBM can handle ordinal encoding)
        - Keep propensity_score as a feature — it captures risk information
        - Drop customer_id (not a feature) and revenue_at_risk (leaks outcome)
        - Drop tenure_bucket (derived from tenure, would be redundant)

    WHAT YOU LEARN:
        Feature selection matters. Including the wrong features (like
        revenue_at_risk which directly depends on churn_binary) would
        cause DATA LEAKAGE — your model sees the answer in the features.
        This is a common mistake in portfolio projects.
    """
    print("[INFO] Loading modeling dataset from Phase 1...")
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df)} rows × {df.shape[1]} columns")

    # ── Define feature columns ──
    # These are the features both models will use for prediction.
    # They must NOT include the treatment indicator or the outcome.
    drop_cols = [
        "customer_id",       # Identifier, not a feature
        "treatment",         # This is what splits our two models
        "churn_binary",      # This is the outcome (Y) we're predicting
        "revenue_at_risk",   # Derived from churn_binary — DATA LEAKAGE
        "tenure_bucket",     # Derived from tenure_months — redundant
    ]

    feature_cols = [c for c in df.columns if c not in drop_cols]

    # ── Encode categorical features ──
    label_encoders = {}
    df_encoded = df.copy()

    for col in df_encoded[feature_cols].select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
        print(f"  Encoded: {col} → {le.classes_.tolist()}")

    X = df_encoded[feature_cols].values
    y = df_encoded["churn_binary"].values
    treatment = df_encoded["treatment"].values

    print(f"\n[INFO] Features: {len(feature_cols)}")
    print(f"[INFO] Treatment group: {treatment.sum():,} customers")
    print(f"[INFO] Control group:   {(1 - treatment).sum():,} customers")

    return X, y, treatment, feature_cols, label_encoders, df


def train_t_learner(X: np.ndarray, y: np.ndarray, treatment: np.ndarray,
                    feature_names: list) -> tuple:
    """
    Train the T-Learner: two separate models on treatment and control groups.

    CAUSAL MATH — THE T-LEARNER ARCHITECTURE:

        Step 1: Split data by treatment status
            Treatment group: {(Xᵢ, Yᵢ) : Tᵢ = 1}
            Control group:   {(Xᵢ, Yᵢ) : Tᵢ = 0}

        Step 2: Train separate models
            Model 1: μ₁(x) = E[Y | X=x, T=1]  (predicts churn IF treated)
            Model 0: μ₀(x) = E[Y | X=x, T=0]  (predicts churn IF not treated)

        Step 3: Estimate uplift for ANY customer
            τ̂(x) = μ₀(x) - μ₁(x)
            Positive τ → offer REDUCES churn → Persuadable
            Negative τ → offer INCREASES churn → Sleeping Dog

    WHY GRADIENT BOOSTING (not XGBoost)?
        scikit-learn's GradientBoostingClassifier is used here because:
        1. It's built into sklearn — no extra dependency issues
        2. For 7,000 rows, it's more than sufficient
        3. The focus of this project is CAUSAL METHODOLOGY, not model tuning

        In production with millions of rows, you'd use XGBoost or LightGBM
        for speed. The causal logic is identical.

    IMPORTANT NOTE ON SAMPLE SIZE:
        The T-Learner splits data in half (roughly). Each model trains on
        a SUBSET. With 7,043 total customers and ~30% treatment rate:
          - Model 1 trains on ~2,100 customers (treatment)
          - Model 0 trains on ~4,900 customers (control)

        This is a real limitation. The X-Learner (Script 2) addresses
        this by using cross-imputation to leverage ALL the data.
    """
    print("\n" + "=" * 60)
    print("TRAINING T-LEARNER")
    print("=" * 60)

    # ── Split data by treatment status ──
    X_treated = X[treatment == 1]
    y_treated = y[treatment == 1]
    X_control = X[treatment == 0]
    y_control = y[treatment == 0]

    print(f"\n  Treatment group: {len(X_treated):,} samples, churn rate: {y_treated.mean():.1%}")
    print(f"  Control group:   {len(X_control):,} samples, churn rate: {y_control.mean():.1%}")

    # ── Model hyperparameters ──
    # Conservative settings to avoid overfitting on small subsets.
    # In a real project, you'd tune these with cross-validation.
    params = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "min_samples_leaf": 20,
        "random_state": RANDOM_SEED,
    }

    # ── Train Model 0: Control group ──
    # This model learns: "What is the churn probability for customers
    # who did NOT receive a marketing offer?"
    print("\n  Training Model 0 (Control)...")
    model_control = GradientBoostingClassifier(**params)
    model_control.fit(X_control, y_control)

    # Cross-validation on control group
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scores_control = cross_val_score(model_control, X_control, y_control,
                                      cv=cv, scoring="roc_auc")
    print(f"  Model 0 CV AUC: {scores_control.mean():.4f} (±{scores_control.std():.4f})")

    # ── Train Model 1: Treatment group ──
    # This model learns: "What is the churn probability for customers
    # who DID receive a marketing offer?"
    print("\n  Training Model 1 (Treatment)...")
    model_treatment = GradientBoostingClassifier(**params)
    model_treatment.fit(X_treated, y_treated)

    # Cross-validation on treatment group
    scores_treatment = cross_val_score(model_treatment, X_treated, y_treated,
                                        cv=cv, scoring="roc_auc")
    print(f"  Model 1 CV AUC: {scores_treatment.mean():.4f} (±{scores_treatment.std():.4f})")

    # ── Feature importance comparison ──
    # Comparing what each model considers important reveals how
    # treatment changes the drivers of churn.
    print("\n  Top 5 Features by Importance:")
    print(f"  {'Feature':<25} {'Control Model':>15} {'Treatment Model':>15}")
    print(f"  {'-'*55}")

    imp_control = model_control.feature_importances_
    imp_treatment = model_treatment.feature_importances_

    # Sort by control model importance
    top_idx = np.argsort(imp_control)[::-1][:5]
    for idx in top_idx:
        print(f"  {feature_names[idx]:<25} {imp_control[idx]:>15.4f} {imp_treatment[idx]:>15.4f}")

    return model_control, model_treatment


def compute_uplift(model_control, model_treatment, X: np.ndarray) -> np.ndarray:
    """
    Compute the Conditional Average Treatment Effect (CATE) for each customer.

    THE KEY FORMULA:
        τ̂(Xᵢ) = μ₀(Xᵢ) - μ₁(Xᵢ)

        Where:
        μ₀(Xᵢ) = P(churn | Xᵢ, not treated)  → Model 0's prediction
        μ₁(Xᵢ) = P(churn | Xᵢ, treated)      → Model 1's prediction

        We use predict_proba[:, 1] to get churn PROBABILITIES, not classes.
        This gives us a continuous uplift score, not just binary.

    INTERPRETATION:
        τ̂ = +0.30 → This customer's churn probability drops by 30pp with the offer
        τ̂ =  0.00 → The offer has no effect on this customer
        τ̂ = -0.15 → The offer INCREASES this customer's churn by 15pp (Sleeping Dog!)

    WHY μ₀ - μ₁ AND NOT μ₁ - μ₀?
        Because Y=1 means churn (a BAD outcome). We want positive uplift to
        mean the offer HELPS (reduces churn). Since lower churn probability
        is better, we compute: uplift = P(churn without offer) - P(churn with offer).
        If the offer reduces churn, μ₀ > μ₁, so uplift is positive.
    """
    print("\n" + "=" * 60)
    print("COMPUTING UPLIFT SCORES (CATE)")
    print("=" * 60)

    # ── Predict churn probabilities under both scenarios ──
    # For EVERY customer, we ask both models:
    # "What would happen to this person WITH and WITHOUT the offer?"
    mu_0 = model_control.predict_proba(X)[:, 1]     # P(churn | no offer)
    mu_1 = model_treatment.predict_proba(X)[:, 1]    # P(churn | offer)

    # ── CATE (uplift) ──
    uplift = mu_0 - mu_1

    print(f"\n  Predictions (μ₀ — Control model):")
    print(f"    Mean P(churn | no offer): {mu_0.mean():.4f}")
    print(f"    Range: [{mu_0.min():.4f}, {mu_0.max():.4f}]")

    print(f"\n  Predictions (μ₁ — Treatment model):")
    print(f"    Mean P(churn | offer):    {mu_1.mean():.4f}")
    print(f"    Range: [{mu_1.min():.4f}, {mu_1.max():.4f}]")

    print(f"\n  Uplift τ̂(X) = μ₀(X) - μ₁(X):")
    print(f"    Mean uplift:  {uplift.mean():.4f}")
    print(f"    Median:       {np.median(uplift):.4f}")
    print(f"    Range:        [{uplift.min():.4f}, {uplift.max():.4f}]")
    print(f"    Std dev:      {uplift.std():.4f}")
    print(f"    Positive (offer helps): {(uplift > 0).sum():,} ({(uplift > 0).mean():.1%})")
    print(f"    Negative (offer hurts): {(uplift < 0).sum():,} ({(uplift < 0).mean():.1%})")

    return uplift, mu_0, mu_1


def segment_customers(df: pd.DataFrame, uplift: np.ndarray,
                      mu_0: np.ndarray, mu_1: np.ndarray) -> pd.DataFrame:
    """
    Classify each customer into one of the 4 uplift quadrants.

    SEGMENTATION LOGIC:
        We use TWO dimensions to classify:
        1. Uplift score (τ): Does the offer help or hurt?
        2. Baseline churn risk (μ₀): Is this customer at risk without intervention?

        Thresholds:
        - uplift_threshold = 0.01 (minimum meaningful effect)
        - risk_threshold = median of μ₀ (splits high vs low baseline risk)

        Quadrant mapping:
        ┌─────────────────┬──────────────────────────────────┐
        │                 │        Baseline Churn Risk        │
        │                 │    HIGH (μ₀ > median)  │   LOW    │
        ├─────────────────┼──────────────────────────────────┤
        │ Uplift > 0.01   │    PERSUADABLE ★       │ SURE     │
        │                 │    (target these!)     │ THING    │
        ├─────────────────┼──────────────────────────────────┤
        │ Uplift < -0.01  │    LOST CAUSE          │ SLEEPING │
        │                 │                        │ DOG ⚠    │
        ├─────────────────┼──────────────────────────────────┤
        │ |Uplift| ≤ 0.01 │    LOST CAUSE          │ SURE     │
        │                 │                        │ THING    │
        └─────────────────┴──────────────────────────────────┘

    WHY SLEEPING DOGS MATTER:
        These customers would STAY if left alone but CHURN if contacted.
        Every dollar spent contacting them is worse than wasted — it
        actively destroys value. Identifying them is as important as
        finding Persuadables. Most churn models completely miss this.
    """
    print("\n" + "=" * 60)
    print("SEGMENTING INTO UPLIFT QUADRANTS")
    print("=" * 60)

    df = df.copy()
    df["uplift_score"] = uplift
    df["mu_0_control"] = mu_0
    df["mu_1_treatment"] = mu_1

    # ── Define thresholds ──
    uplift_threshold = 0.01
    risk_threshold = np.median(mu_0)

    print(f"\n  Thresholds:")
    print(f"    Uplift threshold: ±{uplift_threshold}")
    print(f"    Risk threshold (median μ₀): {risk_threshold:.4f}")

    # ── Assign quadrants ──
    conditions = [
        (uplift > uplift_threshold) & (mu_0 > risk_threshold),    # Persuadable
        (uplift > uplift_threshold) & (mu_0 <= risk_threshold),   # Sure Thing
        (uplift < -uplift_threshold),                              # Sleeping Dog
    ]
    choices = ["Persuadable", "Sure Thing", "Sleeping Dog"]

    df["quadrant"] = np.select(conditions, choices, default="Lost Cause")

    # ── Compute Value at Risk for each customer ──
    # This converts uplift from a probability to a DOLLAR amount.
    # value_at_risk = CLV × uplift_score
    # For Persuadables: positive value → potential savings
    # For Sleeping Dogs: negative value → potential damage
    df["value_at_risk"] = (df["clv"] * df["uplift_score"]).round(2)

    # ── Summary statistics ──
    print(f"\n  Quadrant Distribution:")
    print(f"  {'Quadrant':<15} {'Count':>8} {'%':>8} {'Avg Uplift':>12} {'Avg CLV':>10} {'Total Value@Risk':>18}")
    print(f"  {'-'*71}")

    for q in ["Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog"]:
        subset = df[df["quadrant"] == q]
        if len(subset) > 0:
            print(f"  {q:<15} {len(subset):>8,} {len(subset)/len(df)*100:>7.1f}% "
                  f"{subset['uplift_score'].mean():>12.4f} "
                  f"${subset['clv'].mean():>9,.0f} "
                  f"${subset['value_at_risk'].sum():>17,.2f}")

    # ── Key business insight ──
    persuadables = df[df["quadrant"] == "Persuadable"]
    sleeping_dogs = df[df["quadrant"] == "Sleeping Dog"]

    print(f"\n  ★ KEY BUSINESS INSIGHTS:")
    print(f"    Persuadables: {len(persuadables):,} customers with "
          f"${persuadables['value_at_risk'].sum():,.0f} in saveable revenue")
    print(f"    Sleeping Dogs: {len(sleeping_dogs):,} customers — "
          f"contacting them would DESTROY ${abs(sleeping_dogs['value_at_risk'].sum()):,.0f} in value")

    return df


def export_artifacts(model_control, model_treatment, label_encoders,
                     feature_names: list, df: pd.DataFrame) -> None:
    """
    Save all trained models and data for Phase 3 (API) and Phase 4 (Dashboard).

    WHAT WE EXPORT:
        1. model_control.pkl    — Model 0 (predicts churn without offer)
        2. model_treatment.pkl  — Model 1 (predicts churn with offer)
        3. label_encoders.pkl   — For encoding new customer data at inference
        4. feature_names.pkl    — Ordered list of feature columns
        5. scored_customers.csv — All customers with uplift scores and quadrants

    WHY PICKLE?
        pickle serializes Python objects to binary files. It's the standard
        for saving sklearn models. In production, you'd use MLflow or a
        model registry, but pickle is sufficient for this project.
    """
    print("\n" + "=" * 60)
    print("EXPORTING ARTIFACTS")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Save models ──
    artifacts = {
        "model_control.pkl": model_control,
        "model_treatment.pkl": model_treatment,
        "label_encoders.pkl": label_encoders,
        "feature_names.pkl": feature_names,
    }

    for filename, obj in artifacts.items():
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)
        print(f"  [SAVED] {filepath}")

    # ── Save combined model bundle (for Phase 3 API) ──
    bundle = {
        "model_control": model_control,
        "model_treatment": model_treatment,
        "label_encoders": label_encoders,
        "feature_names": feature_names,
        "model_type": "T-Learner",
        "training_date": pd.Timestamp.now().isoformat(),
        "n_training_samples": len(df),
    }
    bundle_path = os.path.join(OUTPUT_DIR, "t_learner_bundle.pkl")
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  [SAVED] {bundle_path}")

    # ── Save scored customers ──
    scored_path = os.path.join(OUTPUT_DIR, "scored_customers.csv")
    df.to_csv(scored_path, index=False)
    print(f"  [SAVED] {scored_path}")

    print(f"\n  All artifacts saved to: {OUTPUT_DIR}")


# ─────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("PHASE 2 | Script 1: T-LEARNER (BASELINE UPLIFT MODEL)")
    print("=" * 60 + "\n")

    # Step 1: Load and prepare data from Phase 1
    X, y, treatment, feature_names, label_encoders, df = load_and_prepare_data(INPUT_PATH)

    # Step 2: Train T-Learner (two models)
    model_control, model_treatment = train_t_learner(X, y, treatment, feature_names)

    # Step 3: Compute uplift scores for ALL customers
    uplift, mu_0, mu_1 = compute_uplift(model_control, model_treatment, X)

    # Step 4: Segment into quadrants
    df_scored = segment_customers(df, uplift, mu_0, mu_1)

    # Step 5: Export artifacts
    export_artifacts(model_control, model_treatment, label_encoders,
                     feature_names, df_scored)

    print(f"\n{'=' * 60}")
    print("Phase 2, Script 1 (T-Learner) COMPLETE.")
    print("Next: Script 2 (X-Learner) for method comparison.")
    print(f"{'=' * 60}")

    return df_scored


if __name__ == "__main__":
    main()