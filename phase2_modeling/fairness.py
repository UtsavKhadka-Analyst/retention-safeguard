"""
=============================================================================
PHASE 2 | Script 5 of 5: fairness.py
=============================================================================
Proactive Retention & Revenue Safeguard System
Author: Utsav Khadka (github.com/UtsavKhadka-Analyst)

PURPOSE:
    Analyze whether uplift scores are fair across demographic groups.
    This is increasingly REQUIRED in data science interviews (2025+).

WHY FAIRNESS IN UPLIFT MODELING?
    If your uplift model systematically assigns higher scores to
    one demographic group, your marketing campaign will disproportionately
    target (or ignore) that group. This has real consequences:

    - If "Persuadable" is predominantly male, women get ignored → unequal service
    - If "Sleeping Dog" is predominantly senior citizens, you systematically
      avoid contacting them → potential age discrimination
    - If high-CLV Persuadables are all from one group, marketing dollars
      flow disproportionately → equity concerns

    Even with synthetic data, demonstrating that you THINK about fairness
    signals senior-level awareness that junior candidates lack.

METRICS WE USE:
    1. DEMOGRAPHIC PARITY:
       Are the quadrant distributions similar across groups?
       |P(Persuadable | male) - P(Persuadable | female)| < threshold

    2. EQUALIZED UPLIFT:
       Are the mean uplift scores similar across groups?
       |mean_uplift(male) - mean_uplift(female)| < threshold

    3. PROPORTIONAL REPRESENTATION:
       Is each group represented in the Persuadable segment proportionally
       to their representation in the overall population?

INTERVIEW PREP:
    Q: "What fairness considerations apply to your uplift model?"
    A: "If the model disproportionately classifies one demographic as
       Persuadable, marketing resources flow unequally. I check
       demographic parity across uplift quadrants and flag any segments
       where representation deviates by more than 10 percentage points."
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "artifacts")


def load_scored_data() -> pd.DataFrame:
    """Load the T-Learner scored customer data."""
    path = os.path.join(OUTPUT_DIR, "scored_customers.csv")
    if not os.path.exists(path):
        print(f"[ERROR] Scored data not found: {path}")
        print(f"[HINT]  Run t_learner.py first.")
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def demographic_parity_analysis(df: pd.DataFrame) -> None:
    """
    Check demographic parity across uplift quadrants.

    For each demographic group, compute:
    - What % are classified as Persuadable?
    - What % are classified as Sleeping Dog?
    - Is there significant disparity?

    THRESHOLD: Industry standard is 80% rule (or 0.10 absolute difference).
    If one group's Persuadable rate is less than 80% of another group's
    rate, flag it as a potential fairness concern.
    """
    print("\n" + "=" * 60)
    print("DEMOGRAPHIC PARITY ANALYSIS")
    print("=" * 60)

    # ── Analyze by gender ──
    print("\n  ── Gender Analysis ──")
    _analyze_group(df, "gender", "quadrant")

    # ── Analyze by senior citizen status ──
    print("\n  ── Senior Citizen Analysis ──")
    df["senior_label"] = df["senior_citizen"].map({0: "Non-Senior", 1: "Senior"})
    _analyze_group(df, "senior_label", "quadrant")

    # ── Analyze by partner status ──
    print("\n  ── Partner Status Analysis ──")
    _analyze_group(df, "partner", "quadrant")


def _analyze_group(df: pd.DataFrame, group_col: str, quadrant_col: str) -> None:
    """
    Analyze quadrant distribution for a specific demographic group.
    """
    groups = df[group_col].unique()

    print(f"\n  {'Group':<20}", end="")
    for q in ["Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog"]:
        print(f" {q:>14}", end="")
    print(f" {'Avg Uplift':>12} {'Avg CLV':>10}")
    print(f"  {'-'*90}")

    group_persuadable_rates = {}

    for group in sorted(groups):
        subset = df[df[group_col] == group]
        n = len(subset)

        rates = {}
        for q in ["Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog"]:
            q_count = (subset[quadrant_col] == q).sum()
            rates[q] = q_count / n * 100

        avg_uplift = subset["uplift_score"].mean()
        avg_clv = subset["clv"].mean()

        group_persuadable_rates[group] = rates["Persuadable"]

        print(f"  {str(group):<20}", end="")
        for q in ["Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog"]:
            print(f" {rates[q]:>13.1f}%", end="")
        print(f" {avg_uplift:>12.4f} ${avg_clv:>9,.0f}")

    # ── Parity check ──
    if len(group_persuadable_rates) >= 2:
        rates = list(group_persuadable_rates.values())
        max_rate = max(rates)
        min_rate = min(rates)

        if max_rate > 0:
            ratio = min_rate / max_rate
            diff = abs(max_rate - min_rate)

            print(f"\n  Parity Check:")
            print(f"    Max Persuadable rate: {max_rate:.1f}%")
            print(f"    Min Persuadable rate: {min_rate:.1f}%")
            print(f"    Absolute difference:  {diff:.1f} pp")
            print(f"    Ratio (min/max):      {ratio:.3f}")

            if ratio >= 0.80 and diff < 10:
                print(f"    → PASS: Demographic parity satisfied (ratio ≥ 0.80) ✓")
            else:
                print(f"    → FLAG: Potential disparity detected (ratio < 0.80 or diff ≥ 10pp) ⚠")
                print(f"    → This should be investigated before deployment.")


def equalized_uplift_analysis(df: pd.DataFrame) -> None:
    """
    Check whether mean uplift scores are similar across demographics.

    If one group systematically gets higher uplift scores, the model
    predicts that group is more "saveable" — which may reflect real
    heterogeneity or may reflect model bias.
    """
    print("\n" + "=" * 60)
    print("EQUALIZED UPLIFT ANALYSIS")
    print("=" * 60)

    demographic_cols = ["gender", "senior_label", "partner"]

    for col in demographic_cols:
        if col not in df.columns:
            continue

        print(f"\n  ── {col} ──")
        groups = df.groupby(col)["uplift_score"].agg(["mean", "median", "std", "count"])
        groups.columns = ["Mean Uplift", "Median Uplift", "Std Uplift", "Count"]

        print(groups.to_string())

        # Check if difference is statistically meaningful
        group_means = groups["Mean Uplift"].values
        if len(group_means) >= 2:
            max_diff = max(group_means) - min(group_means)
            pooled_std = df["uplift_score"].std()
            effect_size = max_diff / pooled_std if pooled_std > 0 else 0

            print(f"\n    Max difference in means: {max_diff:.4f}")
            print(f"    Effect size (Cohen's d): {effect_size:.4f}")
            if effect_size < 0.20:
                print(f"    → Negligible effect size (< 0.20) ✓")
            elif effect_size < 0.50:
                print(f"    → Small effect size — worth monitoring")
            else:
                print(f"    → Medium/large effect size — investigate ⚠")


def value_distribution_analysis(df: pd.DataFrame) -> None:
    """
    Check how revenue-at-risk and CLV distribute across demographics
    within the Persuadable segment.

    WHY THIS MATTERS:
        Even if the NUMBER of Persuadables is balanced across groups,
        the VALUE might not be. If high-CLV Persuadables are all from
        one group, marketing dollars flow disproportionately.
    """
    print("\n" + "=" * 60)
    print("VALUE DISTRIBUTION IN PERSUADABLE SEGMENT")
    print("=" * 60)

    persuadables = df[df["quadrant"] == "Persuadable"]

    if len(persuadables) == 0:
        print("  No Persuadable customers found.")
        return

    print(f"\n  Total Persuadables: {len(persuadables):,}")
    print(f"  Total saveable value: ${persuadables['value_at_risk'].sum():,.2f}")

    for col in ["gender", "senior_label", "partner"]:
        if col not in persuadables.columns:
            continue

        print(f"\n  ── {col} (within Persuadables) ──")
        group_stats = persuadables.groupby(col).agg({
            "customer_id": "count",
            "clv": "mean",
            "value_at_risk": ["sum", "mean"],
            "uplift_score": "mean",
        }).round(2)

        # Flatten multi-level columns
        group_stats.columns = ["Count", "Avg CLV", "Total Value@Risk",
                                "Avg Value@Risk", "Avg Uplift"]

        # Add percentage
        group_stats["% of Persuadables"] = (
            group_stats["Count"] / len(persuadables) * 100
        ).round(1)

        # Add population percentage for comparison
        pop_pcts = df.groupby(col)["customer_id"].count() / len(df) * 100
        group_stats["% of Population"] = pop_pcts.round(1)

        print(group_stats.to_string())

        # Proportional representation check
        for group in group_stats.index:
            pop_pct = group_stats.loc[group, "% of Population"]
            pers_pct = group_stats.loc[group, "% of Persuadables"]
            diff = pers_pct - pop_pct

            if abs(diff) > 5:
                print(f"\n    ⚠ {group}: {pers_pct:.1f}% of Persuadables vs "
                      f"{pop_pct:.1f}% of population (diff: {diff:+.1f}pp)")


def generate_fairness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a summary fairness report for documentation."""
    print("\n" + "=" * 60)
    print("FAIRNESS SUMMARY REPORT")
    print("=" * 60)

    rows = []
    for col in ["gender", "senior_label", "partner"]:
        if col not in df.columns:
            continue
        for group in df[col].unique():
            subset = df[df[col] == group]
            rows.append({
                "demographic": col,
                "group": group,
                "n_customers": len(subset),
                "pct_persuadable": round((subset["quadrant"] == "Persuadable").mean() * 100, 1),
                "pct_sleeping_dog": round((subset["quadrant"] == "Sleeping Dog").mean() * 100, 1),
                "mean_uplift": round(subset["uplift_score"].mean(), 4),
                "mean_clv": round(subset["clv"].mean(), 2),
                "total_value_at_risk": round(subset["value_at_risk"].sum(), 2),
            })

    report = pd.DataFrame(rows)
    print(report.to_string(index=False))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "fairness_report.csv")
    report.to_csv(path, index=False)
    print(f"\n  [SAVED] {path}")

    return report


# ─────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("PHASE 2 | Script 5: FAIRNESS ANALYSIS")
    print("=" * 60 + "\n")

    # Load scored data
    df = load_scored_data()

    # Run all fairness checks
    demographic_parity_analysis(df)
    equalized_uplift_analysis(df)
    value_distribution_analysis(df)

    # Generate summary report
    report = generate_fairness_report(df)

    print(f"\n{'=' * 60}")
    print("PHASE 2 COMPLETE — ALL 5 SCRIPTS FINISHED")
    print(f"{'=' * 60}")
    print(f"\n  Artifacts generated:")
    print(f"    → t_learner_bundle.pkl     (T-Learner models)")
    print(f"    → x_learner_bundle.pkl     (X-Learner models)")
    print(f"    → scored_customers.csv     (T-Learner scores)")
    print(f"    → scored_customers_xl.csv  (X-Learner scores)")
    print(f"    → model_comparison_report.csv")
    print(f"    → fairness_report.csv")
    print(f"\n  Next: Phase 3 — FastAPI Backend")
    print(f"{'=' * 60}")

    return report


if __name__ == "__main__":
    main()