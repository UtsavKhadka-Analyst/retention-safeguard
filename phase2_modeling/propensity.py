"""
=============================================================================
PHASE 2 | Script 3 of 5: propensity.py
=============================================================================
Proactive Retention & Revenue Safeguard System
Author: Utsav Khadka (github.com/UtsavKhadka-Analyst)

PURPOSE:
    Estimate propensity scores and apply Inverse Probability Weighting (IPW)
    to correct for selection bias in treatment assignment.

WHY IPW?
    In Phase 1, we intentionally assigned treatment non-randomly.
    High-risk customers were more likely to receive offers.
    This means the treatment and control groups are NOT comparable.

    IPW creates a "pseudo-population" where treatment is effectively
    random by reweighting observations:

        Treated customer with weight:  1 / e(x)
        Control customer with weight:  1 / (1 - e(x))

    INTUITION:
        A treated customer with propensity 0.90 gets weight 1/0.90 ≈ 1.1
        → "You were expected to be treated, nothing surprising, low weight"

        A treated customer with propensity 0.10 gets weight 1/0.10 = 10
        → "You were NOT expected to be treated, very informative, high weight"

        This up-weights "surprising" observations and down-weights
        "expected" observations, balancing the groups.


=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import os
import warnings

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE1_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "phase1_data")
INPUT_PATH = os.path.join(PHASE1_DIR, "modeling_dataset.csv")


def load_data(path: str) -> tuple:
    """Load and encode data for propensity modeling."""
    print("[INFO] Loading data for propensity estimation...")
    df = pd.read_csv(path)

    # Features for propensity model (predicting treatment assignment)
    # IMPORTANT: Do NOT include the outcome (churn_binary) as a feature.
    # The propensity score models treatment assignment from PRE-treatment
    # covariates only. Including the outcome would be circular.
    feature_cols = [
        "tenure_months", "monthly_charges", "total_charges",
        "senior_citizen", "contract_type", "internet_service",
        "payment_method", "partner", "dependents", "paperless_billing",
        "num_premium_services", "clv"
    ]

    df_enc = df.copy()
    for col in df_enc[feature_cols].select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))

    X = df_enc[feature_cols].values
    treatment = df_enc["treatment"].values
    y = df_enc["churn_binary"].values

    return X, treatment, y, feature_cols, df


def estimate_propensity_scores(X: np.ndarray, treatment: np.ndarray,
                                feature_names: list) -> np.ndarray:
    """
    Estimate propensity scores using two methods and compare.

    PROPENSITY SCORE: e(x) = P(T=1 | X=x)

    We use two models:
    1. Logistic Regression (interpretable, standard in econometrics)
    2. Gradient Boosting (flexible, captures non-linear relationships)

    We evaluate calibration — a well-calibrated propensity model means
    that among customers with e(x) = 0.30, approximately 30% actually
    received treatment. Poor calibration leads to biased IPW estimates.
    """
    print("\n" + "=" * 60)
    print("ESTIMATING PROPENSITY SCORES")
    print("=" * 60)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # ── Method 1: Logistic Regression ──
    print("\n  Method 1: Logistic Regression")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    lr.fit(X, treatment)
    ps_lr = lr.predict_proba(X)[:, 1]
    auc_lr = cross_val_score(lr, X, treatment, cv=cv, scoring="roc_auc")
    print(f"    AUC: {auc_lr.mean():.4f} (±{auc_lr.std():.4f})")
    print(f"    Score range: [{ps_lr.min():.4f}, {ps_lr.max():.4f}]")

    # ── Method 2: Gradient Boosting ──
    print("\n  Method 2: Gradient Boosting")
    gbm = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        min_samples_leaf=30, random_state=RANDOM_SEED
    )
    gbm.fit(X, treatment)
    ps_gbm = gbm.predict_proba(X)[:, 1]
    auc_gbm = cross_val_score(gbm, X, treatment, cv=cv, scoring="roc_auc")
    print(f"    AUC: {auc_gbm.mean():.4f} (±{auc_gbm.std():.4f})")
    print(f"    Score range: [{ps_gbm.min():.4f}, {ps_gbm.max():.4f}]")

    # ── Select best model ──
    if auc_gbm.mean() > auc_lr.mean():
        ps_final = ps_gbm
        best_model = "Gradient Boosting"
    else:
        ps_final = ps_lr
        best_model = "Logistic Regression"

    print(f"\n  Selected: {best_model} (higher AUC)")

    # ── Feature importance for propensity ──
    # Which features most strongly predict treatment assignment?
    # This reveals the selection mechanism.
    print(f"\n  Top features driving treatment assignment:")
    importances = gbm.feature_importances_
    top_idx = np.argsort(importances)[::-1][:5]
    for idx in top_idx:
        print(f"    {feature_names[idx]:<25} {importances[idx]:.4f}")

    return ps_final


def compute_ipw_ate(y: np.ndarray, treatment: np.ndarray,
                    propensity: np.ndarray) -> dict:
    """
    Compute the Average Treatment Effect (ATE) using Inverse Probability Weighting.

    IPW FORMULA:
        ATE = (1/n) Σ [ Tᵢ·Yᵢ/e(Xᵢ) - (1-Tᵢ)·Yᵢ/(1-e(Xᵢ)) ]

    Since Y=1 means churn (bad), we NEGATE to get:
        ATE > 0 means the offer REDUCES churn on average

    THIS IS THE CRITICAL COMPARISON:

        Naive ATE (no correction):
            Simply compare mean(Y | T=1) - mean(Y | T=0)
            This is BIASED because groups aren't comparable.

        IPW ATE (corrected):
            Reweight observations to simulate random assignment.
            This should give a DIFFERENT (and more accurate) estimate.

    If both give the same answer, either:
        a) There's no selection bias (unlikely given our design)
        b) The propensity model is wrong

    INTERVIEW PREP:
        Q: "Walk me through how IPW corrects selection bias."
        A: "IPW reweights each observation by the inverse of its
           probability of being assigned to its actual group.
           This creates a pseudo-population where treatment assignment
           is independent of covariates, allowing valid causal estimation.
           It's analogous to survey weighting for non-response bias."
    """
    print("\n" + "=" * 60)
    print("IPW — AVERAGE TREATMENT EFFECT ESTIMATION")
    print("=" * 60)

    # ── Clip propensity scores to avoid extreme weights ──
    # Without clipping, a customer with e(x) = 0.01 would get weight 100,
    # dominating the entire estimate. This is called "practical positivity violation."
    e = np.clip(propensity, 0.05, 0.95)

    # ── Naive ATE (biased — no correction) ──
    naive_treated_churn = y[treatment == 1].mean()
    naive_control_churn = y[treatment == 0].mean()
    naive_ate = naive_control_churn - naive_treated_churn  # Positive = offer helps

    print(f"\n  NAIVE ESTIMATE (no correction):")
    print(f"    Treated churn rate:  {naive_treated_churn:.4f}")
    print(f"    Control churn rate:  {naive_control_churn:.4f}")
    print(f"    Naive ATE:           {naive_ate:.4f}")
    print(f"    Interpretation:      {'Offer REDUCES churn' if naive_ate > 0 else 'Offer INCREASES churn'}")
    if naive_ate < 0:
        print(f"    ⚠ WARNING: Naive estimate says offer increases churn!")
        print(f"    This is the SELECTION BIAS we intentionally created.")

    # ── IPW ATE (bias-corrected) ──
    # Horvitz-Thompson estimator
    w_treated = treatment / e
    w_control = (1 - treatment) / (1 - e)

    ipw_treated_churn = np.sum(w_treated * y) / np.sum(w_treated)
    ipw_control_churn = np.sum(w_control * y) / np.sum(w_control)
    ipw_ate = ipw_control_churn - ipw_treated_churn  # Positive = offer helps

    print(f"\n  IPW ESTIMATE (bias-corrected):")
    print(f"    Weighted treated churn: {ipw_treated_churn:.4f}")
    print(f"    Weighted control churn: {ipw_control_churn:.4f}")
    print(f"    IPW ATE:                {ipw_ate:.4f}")
    print(f"    Interpretation:         {'Offer REDUCES churn' if ipw_ate > 0 else 'Offer INCREASES churn'}")

    # ── Comparison ──
    print(f"\n  COMPARISON:")
    print(f"    Naive ATE:  {naive_ate:+.4f}")
    print(f"    IPW ATE:    {ipw_ate:+.4f}")
    print(f"    Difference: {abs(ipw_ate - naive_ate):.4f}")
    print(f"    → IPW correction shifted the estimate by {abs(ipw_ate - naive_ate):.4f}")

    # ── Weight diagnostics ──
    all_weights = np.where(treatment == 1, 1/e, 1/(1-e))
    print(f"\n  Weight diagnostics:")
    print(f"    Mean weight:   {all_weights.mean():.2f}")
    print(f"    Max weight:    {all_weights.max():.2f}")
    print(f"    Min weight:    {all_weights.min():.2f}")
    print(f"    Effective n:   {(np.sum(all_weights)**2 / np.sum(all_weights**2)):.0f} "
          f"(of {len(y)} actual)")

    return {
        "naive_ate": naive_ate,
        "ipw_ate": ipw_ate,
        "naive_treated_churn": naive_treated_churn,
        "naive_control_churn": naive_control_churn,
        "ipw_treated_churn": ipw_treated_churn,
        "ipw_control_churn": ipw_control_churn,
    }


def covariate_balance_check(X: np.ndarray, treatment: np.ndarray,
                             propensity: np.ndarray, feature_names: list) -> None:
    """
    Check whether IPW achieves covariate balance between groups.

    WHAT IS COVARIATE BALANCE?
        After IPW reweighting, the distribution of features should be
        SIMILAR between treatment and control groups. If they're not,
        the propensity model is inadequate.

    METRIC: Standardized Mean Difference (SMD)
        SMD = |mean_treated - mean_control| / pooled_std
        Rule of thumb: SMD < 0.10 is acceptable balance

    This is a CRITICAL diagnostic that most portfolio projects skip.
    Including it demonstrates methodological rigor.
    """
    print("\n" + "=" * 60)
    print("COVARIATE BALANCE CHECK (BEFORE vs AFTER IPW)")
    print("=" * 60)

    e = np.clip(propensity, 0.05, 0.95)
    w_treated = 1 / e
    w_control = 1 / (1 - e)

    print(f"\n  {'Feature':<25} {'SMD (Raw)':>12} {'SMD (IPW)':>12} {'Balanced?':>10}")
    print(f"  {'-'*60}")

    n_balanced_before = 0
    n_balanced_after = 0

    for i, fname in enumerate(feature_names):
        x = X[:, i]
        x_t = x[treatment == 1]
        x_c = x[treatment == 0]

        # Raw SMD (before IPW)
        pooled_std = np.sqrt((x_t.var() + x_c.var()) / 2)
        if pooled_std > 0:
            smd_raw = abs(x_t.mean() - x_c.mean()) / pooled_std
        else:
            smd_raw = 0.0

        # IPW-weighted SMD
        wt = np.where(treatment == 1, w_treated, w_control)
        w_mean_t = np.average(x[treatment == 1], weights=w_treated[treatment == 1])
        w_mean_c = np.average(x[treatment == 0], weights=w_control[treatment == 0])
        if pooled_std > 0:
            smd_ipw = abs(w_mean_t - w_mean_c) / pooled_std
        else:
            smd_ipw = 0.0

        balanced = "✓" if smd_ipw < 0.10 else "✗"
        if smd_raw < 0.10:
            n_balanced_before += 1
        if smd_ipw < 0.10:
            n_balanced_after += 1

        print(f"  {fname:<25} {smd_raw:>12.4f} {smd_ipw:>12.4f} {balanced:>10}")

    print(f"\n  Summary:")
    print(f"    Balanced features (SMD < 0.10) before IPW: "
          f"{n_balanced_before}/{len(feature_names)}")
    print(f"    Balanced features (SMD < 0.10) after IPW:  "
          f"{n_balanced_after}/{len(feature_names)}")

    if n_balanced_after > n_balanced_before:
        print(f"    → IPW IMPROVED covariate balance ✓")
    else:
        print(f"    → IPW did not improve balance — propensity model may need refinement")


# ─────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("PHASE 2 | Script 3: PROPENSITY SCORES & IPW CORRECTION")
    print("=" * 60 + "\n")

    # Step 1: Load data
    X, treatment, y, feature_names, df = load_data(INPUT_PATH)

    # Step 2: Estimate propensity scores
    propensity = estimate_propensity_scores(X, treatment, feature_names)

    # Step 3: Compute ATE with and without IPW correction
    ate_results = compute_ipw_ate(y, treatment, propensity)

    # Step 4: Check covariate balance
    covariate_balance_check(X, treatment, propensity, feature_names)

    print(f"\n{'=' * 60}")
    print("Phase 2, Script 3 (Propensity & IPW) COMPLETE.")
    print("Next: Script 4 (evaluation.py) for Qini curves.")
    print(f"{'=' * 60}")

    return ate_results


if __name__ == "__main__":
    main()