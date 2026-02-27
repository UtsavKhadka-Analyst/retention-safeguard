"""
=============================================================================
PHASE 2 | Script 4 of 5: evaluation.py
=============================================================================
Proactive Retention & Revenue Safeguard System
Author: Utsav Khadka (github.com/UtsavKhadka-Analyst)

PURPOSE:
    Evaluate uplift models using the CORRECT metrics for causal models.
    Standard ML metrics (AUC, F1) do NOT apply to uplift models.

WHY NOT AUC?
    AUC measures how well a model RANKS customers by churn probability.
    But we don't care about churn probability — we care about TREATMENT
    EFFECT. A customer with 90% churn probability might have zero uplift
    (Lost Cause), while a customer with 40% churn might have huge uplift
    (Persuadable).

    Uplift models need UPLIFT-SPECIFIC metrics:

    1. QINI CURVE & COEFFICIENT:
       Plot cumulative uplift (treatment effect captured) as you target
       more customers, sorted by predicted uplift score.
       Qini coefficient = area between your curve and the random baseline.
       Higher = better.

    2. UPLIFT@K:
       "If I target the top K% of customers by uplift score, what
       percentage of the total treatment effect do I capture?"
       This directly answers the budget allocation question.

    3. CUMULATIVE GAIN CHART:
       Shows cumulative difference in outcomes between treatment and
       control as you move through uplift-ranked customers.

INTERVIEW PREP:
    Q: "How do you evaluate an uplift model?"
    A: "Not with AUC — that measures prediction, not causal effect.
       I use Qini curves to visualize targeting efficiency, Qini
       coefficient as a single-number summary, and uplift@k to answer
       'how much effect do we capture by targeting the top k%?'"
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "artifacts")


def compute_qini_curve(y: np.ndarray, treatment: np.ndarray,
                        uplift_scores: np.ndarray) -> dict:
    """
    Compute the Qini curve data points.

    THE QINI CURVE — HOW IT WORKS:

        1. Sort all customers by predicted uplift score (highest first)
        2. Walk through the sorted list, targeting one more customer at each step
        3. At each step k, compute:
           Qini(k) = (n_treated_retained(k) / n_treated(k)) * n(k)
                    - (n_control_retained(k) / n_control(k)) * n(k)

        Simplified version (what we compute):
        At each fraction φ of customers targeted:
            qini(φ) = (retained_treated / n_treated - retained_control / n_control) × φ

        INTERPRETATION:
            The curve shows how many ADDITIONAL retentions you get by
            targeting customers in uplift-score order vs. random order.
            Higher curve = better model.

    THE QINI COEFFICIENT:
        Area between your Qini curve and the random targeting diagonal.
        Analogous to AUC for classification, but for uplift.
        Range: can be negative (worse than random) to positive.

    RANDOM BASELINE:
        If you target customers randomly, the Qini curve is a straight line
        from (0,0) to (1, overall_uplift). Your model should beat this.
    """
    # ── Sort by uplift score (descending — target best first) ──
    sorted_idx = np.argsort(-uplift_scores)
    y_sorted = y[sorted_idx]
    t_sorted = treatment[sorted_idx]

    n = len(y)
    n_treated_total = treatment.sum()
    n_control_total = (1 - treatment).sum()

    # ── Compute cumulative Qini values ──
    fractions = []
    qini_values = []

    # We use Y=1 as churn. Retention = 1 - churn.
    # Uplift on retention: higher retention in treatment = good.
    for k in range(1, n + 1, max(1, n // 100)):  # Sample 100 points
        y_k = y_sorted[:k]
        t_k = t_sorted[:k]

        n_t_k = t_k.sum()
        n_c_k = (1 - t_k).sum()

        if n_t_k > 0 and n_c_k > 0:
            # Churn rates in top-k
            churn_t = y_k[t_k == 1].sum() / n_t_k
            churn_c = y_k[t_k == 0].sum() / n_c_k

            # Uplift on retention (positive = offer helps retain)
            retention_uplift = churn_c - churn_t

            # Qini value: uplift × fraction targeted
            fraction = k / n
            qini_val = retention_uplift * fraction

            fractions.append(fraction)
            qini_values.append(qini_val)

    fractions = np.array(fractions)
    qini_values = np.array(qini_values)

    # ── Random baseline ──
    overall_uplift = (y[treatment == 0].mean() - y[treatment == 1].mean())
    random_qini = fractions * overall_uplift

    # ── Qini coefficient (area between model curve and random) ──
    qini_coefficient = np.sum((qini_values[1:] - random_qini[1:] + qini_values[:-1] - random_qini[:-1]) / 2 * np.diff(fractions))

    return {
        "fractions": fractions,
        "qini_values": qini_values,
        "random_qini": random_qini,
        "qini_coefficient": qini_coefficient,
        "overall_uplift": overall_uplift,
    }


def compute_uplift_at_k(y: np.ndarray, treatment: np.ndarray,
                         uplift_scores: np.ndarray,
                         k_values: list = [10, 20, 30, 40, 50]) -> dict:
    """
    Compute uplift@k: treatment effect captured by targeting top k%.

    THIS IS THE MOST BUSINESS-RELEVANT METRIC.

    SCENARIO:
        Marketing manager says: "I have budget to target 20% of customers."
        You need to answer: "If you target the top 20% by our uplift model,
        you'll capture X% of the total possible retention uplift."

    FORMULA:
        uplift@k = (effect in top k%) / (total effect in all data)

        Where effect = churn_rate_control - churn_rate_treated
        for the subset of customers in the top k% by uplift score.

    IDEAL RESULT:
        uplift@20 = 60% means targeting the top 20% captures 60% of
        the total possible effect. That's 3x more efficient than random.
    """
    sorted_idx = np.argsort(-uplift_scores)
    y_sorted = y[sorted_idx]
    t_sorted = treatment[sorted_idx]

    n = len(y)

    # Total effect in all data
    total_effect = y[treatment == 0].mean() - y[treatment == 1].mean()

    results = {}

    for k in k_values:
        cutoff = int(n * k / 100)
        y_k = y_sorted[:cutoff]
        t_k = t_sorted[:cutoff]

        n_t = t_k.sum()
        n_c = (1 - t_k).sum()

        if n_t > 0 and n_c > 0 and total_effect != 0:
            effect_k = y_k[t_k == 0].mean() - y_k[t_k == 1].mean()
            capture_pct = (effect_k / total_effect) * (k / 100) * 100
            results[k] = {
                "effect": effect_k,
                "capture_pct": capture_pct,
                "n_targeted": cutoff,
                "n_treated": int(n_t),
                "n_control": int(n_c),
            }
        else:
            results[k] = {
                "effect": 0.0,
                "capture_pct": 0.0,
                "n_targeted": cutoff,
                "n_treated": int(n_t) if n_t else 0,
                "n_control": int(n_c) if n_c else 0,
            }

    return results


def compute_cumulative_gain(y: np.ndarray, treatment: np.ndarray,
                             uplift_scores: np.ndarray) -> dict:
    """
    Compute cumulative gain chart data.

    CUMULATIVE GAIN shows the running difference between treatment
    and control outcomes as you walk through uplift-ranked customers.

    If the model is good:
    - The curve starts steep (high-uplift customers first)
    - It flattens in the middle (diminishing returns)
    - It may DIP at the end (Sleeping Dogs — negative uplift)

    The DIP is the most interesting part — it proves your model
    identified customers who are harmed by treatment.
    """
    sorted_idx = np.argsort(-uplift_scores)
    y_sorted = y[sorted_idx]
    t_sorted = treatment[sorted_idx]

    n = len(y)
    fractions = []
    cumulative_gains = []

    for k in range(100, n + 1, max(1, n // 100)):
        y_k = y_sorted[:k]
        t_k = t_sorted[:k]

        n_t = t_k.sum()
        n_c = (1 - t_k).sum()

        if n_t > 0 and n_c > 0:
            gain = y_k[t_k == 0].mean() - y_k[t_k == 1].mean()
            fractions.append(k / n)
            cumulative_gains.append(gain)

    return {
        "fractions": np.array(fractions),
        "cumulative_gains": np.array(cumulative_gains),
    }


def compare_models(y: np.ndarray, treatment: np.ndarray,
                    uplift_t: np.ndarray, uplift_x: np.ndarray) -> None:
    """
    Compare T-Learner and X-Learner using all evaluation metrics.

    This is the BENCHMARKING section that turns the project from
    "I implemented one model" into "I compared methods and can
    discuss trade-offs" — a much stronger interview signal.
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON: T-LEARNER vs X-LEARNER")
    print("=" * 60)

    # ── Qini Coefficients ──
    print("\n  ── Qini Coefficients ──")
    qini_t = compute_qini_curve(y, treatment, uplift_t)
    qini_x = compute_qini_curve(y, treatment, uplift_x)

    print(f"    T-Learner Qini coefficient: {qini_t['qini_coefficient']:.6f}")
    print(f"    X-Learner Qini coefficient: {qini_x['qini_coefficient']:.6f}")

    winner = "T-Learner" if qini_t['qini_coefficient'] > qini_x['qini_coefficient'] else "X-Learner"
    print(f"    → Winner: {winner}")

    # ── Uplift@K ──
    print("\n  ── Uplift@K Comparison ──")
    uplift_k_t = compute_uplift_at_k(y, treatment, uplift_t)
    uplift_k_x = compute_uplift_at_k(y, treatment, uplift_x)

    print(f"    {'Top K%':<10} {'T-Learner Effect':>18} {'X-Learner Effect':>18}")
    print(f"    {'-'*46}")
    for k in [10, 20, 30, 40, 50]:
        eff_t = uplift_k_t[k]['effect']
        eff_x = uplift_k_x[k]['effect']
        print(f"    {k}%{'':<8} {eff_t:>18.4f} {eff_x:>18.4f}")

    # ── Cumulative Gain Summary ──
    print("\n  ── Cumulative Gain (first vs last decile) ──")
    gain_t = compute_cumulative_gain(y, treatment, uplift_t)
    gain_x = compute_cumulative_gain(y, treatment, uplift_x)

    if len(gain_t['cumulative_gains']) > 1:
        print(f"    T-Learner: First decile gain = {gain_t['cumulative_gains'][0]:.4f}, "
              f"Last decile gain = {gain_t['cumulative_gains'][-1]:.4f}")
    if len(gain_x['cumulative_gains']) > 1:
        print(f"    X-Learner: First decile gain = {gain_x['cumulative_gains'][0]:.4f}, "
              f"Last decile gain = {gain_x['cumulative_gains'][-1]:.4f}")

    # ── Correlation between methods ──
    correlation = np.corrcoef(uplift_t, uplift_x)[0, 1]
    print(f"\n  ── Score Correlation ──")
    print(f"    Pearson correlation: {correlation:.4f}")
    print(f"    → {'High agreement' if correlation > 0.7 else 'Methods disagree significantly'}")

    # ── Summary recommendation ──
    print(f"\n  ── RECOMMENDATION ──")
    if abs(qini_t['qini_coefficient'] - qini_x['qini_coefficient']) < 0.001:
        print(f"    Both methods perform similarly. T-Learner is simpler → use it.")
        print(f"    The similarity is expected with synthetic data.")
    elif winner == "X-Learner":
        print(f"    X-Learner outperforms. Use X-Learner for deployment.")
        print(f"    Its cross-imputation likely captured treatment heterogeneity better.")
    else:
        print(f"    T-Learner outperforms. Use T-Learner for deployment.")
        print(f"    The simpler model may generalize better on this dataset size.")


def generate_evaluation_report(y, treatment, uplift_t, uplift_x) -> pd.DataFrame:
    """
    Generate a summary report comparing both models.
    This DataFrame will be used in Phase 4 (Dashboard).
    """
    print("\n" + "=" * 60)
    print("GENERATING EVALUATION REPORT")
    print("=" * 60)

    rows = []
    for name, scores in [("T-Learner", uplift_t), ("X-Learner", uplift_x)]:
        qini = compute_qini_curve(y, treatment, scores)
        uplift_k = compute_uplift_at_k(y, treatment, scores)

        rows.append({
            "model": name,
            "qini_coefficient": round(qini["qini_coefficient"], 6),
            "mean_uplift": round(scores.mean(), 4),
            "median_uplift": round(np.median(scores), 4),
            "std_uplift": round(scores.std(), 4),
            "pct_positive_uplift": round((scores > 0).mean() * 100, 1),
            "uplift_at_10": round(uplift_k[10]["effect"], 4),
            "uplift_at_20": round(uplift_k[20]["effect"], 4),
            "uplift_at_50": round(uplift_k[50]["effect"], 4),
        })

    report = pd.DataFrame(rows)

    print(report.to_string(index=False))

    # Save report
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, "model_comparison_report.csv")
    report.to_csv(report_path, index=False)
    print(f"\n  [SAVED] {report_path}")

    return report


# ─────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────
def main():
    """
    Run full evaluation pipeline.
    Requires that t_learner.py and x_learner.py have been run first.
    """
    print("=" * 60)
    print("PHASE 2 | Script 4: UPLIFT MODEL EVALUATION")
    print("=" * 60 + "\n")

    # ── Load scored data from both models ──
    t_path = os.path.join(OUTPUT_DIR, "scored_customers.csv")
    x_path = os.path.join(OUTPUT_DIR, "scored_customers_xl.csv")

    if not os.path.exists(t_path):
        print("[ERROR] T-Learner results not found. Run t_learner.py first.")
        return
    if not os.path.exists(x_path):
        print("[ERROR] X-Learner results not found. Run x_learner.py first.")
        return

    df_t = pd.read_csv(t_path)
    df_x = pd.read_csv(x_path)

    y = df_t["churn_binary"].values
    treatment = df_t["treatment"].values
    uplift_t = df_t["uplift_score"].values
    uplift_x = df_x["uplift_score_xl"].values

    # ── Compare models ──
    compare_models(y, treatment, uplift_t, uplift_x)

    # ── Generate report ──
    report = generate_evaluation_report(y, treatment, uplift_t, uplift_x)

    print(f"\n{'=' * 60}")
    print("Phase 2, Script 4 (Evaluation) COMPLETE.")
    print("Next: Script 5 (fairness.py) for demographic parity check.")
    print(f"{'=' * 60}")

    return report


if __name__ == "__main__":
    main()