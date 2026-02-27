"""
=============================================================================
PHASE 2 | Script 2 of 5: x_learner.py
=============================================================================
Proactive Retention & Revenue Safeguard System
Author: Utsav Khadka (github.com/UtsavKhadka-Analyst)

PURPOSE:
    Implement the X-Learner for uplift estimation.
    This is the ADVANCED method that improves on the T-Learner.

HOW THE X-LEARNER IMPROVES ON THE T-LEARNER:

    T-Learner problem: Each model only sees HALF the data.
    X-Learner solution: Use CROSS-IMPUTATION to leverage ALL data.

    The X-Learner has 3 stages:

    STAGE 1 — Same as T-Learner:
        Train μ₁(x) on treatment group
        Train μ₀(x) on control group

    STAGE 2 — Impute individual treatment effects:
        For treated customers (we observe Y(1), impute Y(0)):
            D¹ᵢ = Yᵢ(1) - μ₀(Xᵢ)
            "Actual outcome minus what control model THINKS would happen"

        For control customers (we observe Y(0), impute Y(1)):
            D⁰ᵢ = μ₁(Xᵢ) - Yᵢ(0)
            "What treatment model THINKS would happen minus actual outcome"

    STAGE 3 — Train two MORE models on these imputed effects:
        τ₁(x) = model trained on {(Xᵢ, D¹ᵢ)} for treated units
        τ₀(x) = model trained on {(Xᵢ, D⁰ᵢ)} for control units

    FINAL — Combine using propensity score as weight:
        τ(x) = e(x) · τ₀(x) + (1 - e(x)) · τ₁(x)

        Where e(x) is the propensity score.
        This weighting is clever: when treatment is rare, we rely more
        on τ₁ (estimated from treated units, who are more informative
        about the treatment effect). When treatment is common, we rely
        more on τ₀.

    WHY THIS IS BETTER:
        1. Every data point is used (no splitting)
        2. Cross-imputation borrows strength across groups
        3. Propensity-weighted combination handles imbalanced groups
        4. Particularly effective when treatment group is small

INTERVIEW PREP:
    Q: "When would you use X-Learner over T-Learner?"
    A: "When treatment and control groups are very imbalanced in size.
       The X-Learner uses cross-imputation so both models benefit from
       the full dataset. It also uses the propensity score to optimally
       weight the two effect estimators."
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
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
PHASE1_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "phase1_data")
INPUT_PATH = os.path.join(PHASE1_DIR, "modeling_dataset.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "artifacts")


def load_and_prepare_data(path: str) -> tuple:
    """
    Load the modeling dataset — same preparation as T-Learner.
    Consistency between methods is critical for fair comparison.
    """
    print("[INFO] Loading modeling dataset from Phase 1...")
    df = pd.read_csv(path)

    drop_cols = [
        "customer_id", "treatment", "churn_binary",
        "revenue_at_risk", "tenure_bucket",
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    label_encoders = {}
    df_encoded = df.copy()
    for col in df_encoded[feature_cols].select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

    X = df_encoded[feature_cols].values
    y = df_encoded["churn_binary"].values
    treatment = df_encoded["treatment"].values
    propensity = df_encoded["propensity_score"].values

    print(f"[INFO] Loaded {len(df)} rows, {len(feature_cols)} features")
    print(f"[INFO] Treatment: {treatment.sum():,} | Control: {(1-treatment).sum():,}")

    return X, y, treatment, propensity, feature_cols, label_encoders, df


def train_x_learner(X: np.ndarray, y: np.ndarray, treatment: np.ndarray,
                    propensity: np.ndarray, feature_names: list) -> tuple:
    """
    Train the X-Learner: 3-stage process with cross-imputation.

    This implementation follows the Künzel et al. (2019) paper:
    "Metalearners for estimating heterogeneous treatment effects
    using machine learning" — PNAS.
    """
    print("\n" + "=" * 60)
    print("TRAINING X-LEARNER (3-STAGE PROCESS)")
    print("=" * 60)

    # ── Split data ──
    X_treated = X[treatment == 1]
    y_treated = y[treatment == 1]
    X_control = X[treatment == 0]
    y_control = y[treatment == 0]

    print(f"\n  Treatment: {len(X_treated):,} | Control: {len(X_control):,}")

    # ── Model parameters (same as T-Learner for fair comparison) ──
    clf_params = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "min_samples_leaf": 20,
        "random_state": RANDOM_SEED,
    }

    # Regressor params for Stage 2 (predicting continuous treatment effects)
    reg_params = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "min_samples_leaf": 20,
        "random_state": RANDOM_SEED,
    }

    # ══════════════════════════════════════════════════════════════════
    # STAGE 1: Train outcome models (identical to T-Learner)
    # ══════════════════════════════════════════════════════════════════
    print("\n  ── Stage 1: Outcome Models ──")

    # Model 0: predict churn for control group
    model_control = GradientBoostingClassifier(**clf_params)
    model_control.fit(X_control, y_control)
    print(f"  Model 0 (Control) trained on {len(X_control):,} samples")

    # Model 1: predict churn for treatment group
    model_treatment = GradientBoostingClassifier(**clf_params)
    model_treatment.fit(X_treated, y_treated)
    print(f"  Model 1 (Treatment) trained on {len(X_treated):,} samples")

    # ══════════════════════════════════════════════════════════════════
    # STAGE 2: Impute individual treatment effects
    # ══════════════════════════════════════════════════════════════════
    print("\n  ── Stage 2: Cross-Imputation ──")

    # For TREATED customers: we observed Y(1), impute Y(0) using control model
    # D¹ᵢ = Yᵢ(1) - μ₀(Xᵢ)
    # This asks: "How much did the actual outcome differ from what the
    # control model predicted would happen without treatment?"
    mu_0_for_treated = model_control.predict_proba(X_treated)[:, 1]
    D_treated = y_treated - mu_0_for_treated  # Negative = offer helped (reduced churn)

    print(f"  D¹ (treated imputed effects): mean={D_treated.mean():.4f}, "
          f"std={D_treated.std():.4f}")

    # For CONTROL customers: we observed Y(0), impute Y(1) using treatment model
    # D⁰ᵢ = μ₁(Xᵢ) - Yᵢ(0)
    # This asks: "How much does the treatment model think would change
    # compared to what actually happened without treatment?"
    mu_1_for_control = model_treatment.predict_proba(X_control)[:, 1]
    D_control = mu_1_for_control - y_control  # Negative = offer would help

    print(f"  D⁰ (control imputed effects): mean={D_control.mean():.4f}, "
          f"std={D_control.std():.4f}")

    # ══════════════════════════════════════════════════════════════════
    # STAGE 3: Train effect models on imputed treatment effects
    # ══════════════════════════════════════════════════════════════════
    print("\n  ── Stage 3: Effect Models ──")

    # τ₁(x): model trained on imputed effects from treated units
    # This is a REGRESSOR because D values are continuous
    tau_model_treated = GradientBoostingRegressor(**reg_params)
    tau_model_treated.fit(X_treated, D_treated)
    print(f"  τ₁ model trained on {len(X_treated):,} imputed effects")

    # τ₀(x): model trained on imputed effects from control units
    tau_model_control = GradientBoostingRegressor(**reg_params)
    tau_model_control.fit(X_control, D_control)
    print(f"  τ₀ model trained on {len(X_control):,} imputed effects")

    return (model_control, model_treatment,
            tau_model_treated, tau_model_control)


def compute_x_learner_uplift(tau_model_treated, tau_model_control,
                              model_control, model_treatment,
                              X: np.ndarray, propensity: np.ndarray) -> np.ndarray:
    """
    Compute final X-Learner uplift by combining τ₁ and τ₀ with propensity weights.

    FINAL FORMULA:
        τ̂(x) = e(x) · τ₀(x) + (1 - e(x)) · τ₁(x)

    Then negate to match our convention (positive = offer helps):
        uplift = -τ̂(x)

    WHY PROPENSITY-WEIGHTED COMBINATION?
        When e(x) is high (customer was likely to be treated):
            → We weight τ₀ more heavily
            → τ₀ was estimated from CONTROL units
            → Control units are more "surprising" (they didn't get treatment
              despite being similar to those who did) → more informative

        When e(x) is low (customer was unlikely to be treated):
            → We weight τ₁ more heavily
            → τ₁ was estimated from TREATED units
            → Treated units are more "surprising" (they got treatment
              despite being unlikely candidates) → more informative

        This optimal weighting is what makes X-Learner superior to
        simple averaging of the two effect estimates.
    """
    print("\n" + "=" * 60)
    print("COMPUTING X-LEARNER UPLIFT")
    print("=" * 60)

    # ── Get effect predictions from both models ──
    tau_1 = tau_model_treated.predict(X)   # Effect estimate from treated data
    tau_0 = tau_model_control.predict(X)   # Effect estimate from control data

    print(f"\n  τ₁ predictions: mean={tau_1.mean():.4f}, range=[{tau_1.min():.4f}, {tau_1.max():.4f}]")
    print(f"  τ₀ predictions: mean={tau_0.mean():.4f}, range=[{tau_0.min():.4f}, {tau_0.max():.4f}]")

    # ── Propensity-weighted combination ──
    # Clip propensity to avoid extreme weights (standard practice)
    e = np.clip(propensity, 0.05, 0.95)

    tau_combined = e * tau_0 + (1 - e) * tau_1

    # ── Negate: our convention is positive uplift = offer reduces churn ──
    uplift = -tau_combined

    print(f"\n  Final X-Learner Uplift:")
    print(f"    Mean:    {uplift.mean():.4f}")
    print(f"    Median:  {np.median(uplift):.4f}")
    print(f"    Range:   [{uplift.min():.4f}, {uplift.max():.4f}]")
    print(f"    Std dev: {uplift.std():.4f}")

    # Also compute control-model predicted churn (for quadrant assignment)
    mu_0 = model_control.predict_proba(X)[:, 1]
    mu_1 = model_treatment.predict_proba(X)[:, 1]

    return uplift, mu_0, mu_1


def segment_customers(df: pd.DataFrame, uplift: np.ndarray,
                      mu_0: np.ndarray, mu_1: np.ndarray) -> pd.DataFrame:
    """
    Segment customers into quadrants using X-Learner uplift scores.
    Same logic as T-Learner segmentation for fair comparison.
    """
    print("\n" + "=" * 60)
    print("SEGMENTING INTO UPLIFT QUADRANTS (X-LEARNER)")
    print("=" * 60)

    df = df.copy()
    df["uplift_score_xl"] = uplift
    df["mu_0_xl"] = mu_0
    df["mu_1_xl"] = mu_1

    uplift_threshold = 0.01
    risk_threshold = np.median(mu_0)

    conditions = [
        (uplift > uplift_threshold) & (mu_0 > risk_threshold),
        (uplift > uplift_threshold) & (mu_0 <= risk_threshold),
        (uplift < -uplift_threshold),
    ]
    choices = ["Persuadable", "Sure Thing", "Sleeping Dog"]
    df["quadrant_xl"] = np.select(conditions, choices, default="Lost Cause")

    df["value_at_risk_xl"] = (df["clv"] * df["uplift_score_xl"]).round(2)

    # ── Summary ──
    print(f"\n  X-Learner Quadrant Distribution:")
    print(f"  {'Quadrant':<15} {'Count':>8} {'%':>8} {'Avg Uplift':>12} {'Total Value@Risk':>18}")
    print(f"  {'-'*61}")

    for q in ["Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog"]:
        subset = df[df["quadrant_xl"] == q]
        if len(subset) > 0:
            print(f"  {q:<15} {len(subset):>8,} {len(subset)/len(df)*100:>7.1f}% "
                  f"{subset['uplift_score_xl'].mean():>12.4f} "
                  f"${subset['value_at_risk_xl'].sum():>17,.2f}")

    return df


def export_artifacts(model_control, model_treatment, tau_model_treated,
                     tau_model_control, label_encoders, feature_names,
                     df: pd.DataFrame) -> None:
    """Save X-Learner model bundle."""
    print("\n" + "=" * 60)
    print("EXPORTING X-LEARNER ARTIFACTS")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    bundle = {
        "model_control": model_control,
        "model_treatment": model_treatment,
        "tau_model_treated": tau_model_treated,
        "tau_model_control": tau_model_control,
        "label_encoders": label_encoders,
        "feature_names": feature_names,
        "model_type": "X-Learner",
        "training_date": pd.Timestamp.now().isoformat(),
        "n_training_samples": len(df),
    }

    bundle_path = os.path.join(OUTPUT_DIR, "x_learner_bundle.pkl")
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  [SAVED] {bundle_path}")

    # ── Save scored customers with both T-Learner and X-Learner scores ──
    scored_path = os.path.join(OUTPUT_DIR, "scored_customers_xl.csv")
    df.to_csv(scored_path, index=False)
    print(f"  [SAVED] {scored_path}")


# ─────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("PHASE 2 | Script 2: X-LEARNER (ADVANCED UPLIFT MODEL)")
    print("=" * 60 + "\n")

    # Step 1: Load data
    X, y, treatment, propensity, feature_names, label_encoders, df = \
        load_and_prepare_data(INPUT_PATH)

    # Step 2: Train X-Learner (3-stage process)
    (model_control, model_treatment,
     tau_model_treated, tau_model_control) = \
        train_x_learner(X, y, treatment, propensity, feature_names)

    # Step 3: Compute X-Learner uplift
    uplift, mu_0, mu_1 = compute_x_learner_uplift(
        tau_model_treated, tau_model_control,
        model_control, model_treatment,
        X, propensity
    )

    # Step 4: Segment customers
    df_scored = segment_customers(df, uplift, mu_0, mu_1)

    # Step 5: Export
    export_artifacts(model_control, model_treatment,
                     tau_model_treated, tau_model_control,
                     label_encoders, feature_names, df_scored)

    print(f"\n{'=' * 60}")
    print("Phase 2, Script 2 (X-Learner) COMPLETE.")
    print("Next: Script 3 (propensity.py) for bias correction.")
    print(f"{'=' * 60}")

    return df_scored


if __name__ == "__main__":
    main()