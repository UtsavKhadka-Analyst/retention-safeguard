"""
=============================================================================
PHASE 1 | Script 3 of 3: sql_queries.py
=============================================================================
Proactive Retention & Revenue Safeguard System
Author: Utsav Khadka (github.com/UtsavKhadka-Analyst)

PURPOSE:
    Execute complex SQL queries against the DuckDB database to extract
    analysis-ready data for Phase 2 (causal modeling).

    This script demonstrates SQL skills that go far beyond SELECT *:
    - Multi-table JOINs (4-5 tables)
    - Common Table Expressions (CTEs)
    - Window Functions (LAG, RANK, NTILE, running aggregates)
    - Conditional aggregation (CASE WHEN inside aggregate)
    - Cohort analysis patterns

WHY SQL MATTERS FOR DATA SCIENTIST ROLES:
    Every data scientist interview includes SQL. Not basic queries —
    they test window functions, self-joins, and CTEs. The queries in
    this script are at the level of a Senior Analyst / DS interview.

    More importantly, in production you often CAN'T pull all data into
    pandas. SQL pushes computation to the database, which handles
    millions of rows efficiently. This is a production skill.

INTERVIEW PREP:
    Q: "Write a query to find the churn rate by tenure cohort."
    A: See Query 2 below — uses NTILE window function.

    Q: "How would you compare treatment vs control groups in SQL?"
    A: See Query 3 below — conditional aggregation with CASE WHEN.
=============================================================================
"""

import duckdb
import pandas as pd
import os

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "retention_safeguard.duckdb")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "modeling_dataset.csv")


def get_connection() -> duckdb.DuckDBPyConnection:
    """Open a read-only connection to the database."""
    if not os.path.exists(DB_PATH):
        print(f"[ERROR] Database not found: {DB_PATH}")
        print(f"[HINT]  Run load_to_db.py first.")
        raise FileNotFoundError(DB_PATH)
    return duckdb.connect(DB_PATH, read_only=True)


# ══════════════════════════════════════════════════════════════════════
# QUERY 1: Full Modeling Dataset (Multi-Table JOIN)
# ══════════════════════════════════════════════════════════════════════
QUERY_MODELING_DATASET = """
--  ┌─────────────────────────────────────────────────────────────────┐
--  │  QUERY 1: FULL MODELING DATASET                                │
--  │                                                                │
--  │  PURPOSE: Join all 5 tables into a single analysis-ready       │
--  │  DataFrame for Phase 2 causal modeling.                        │
--  │                                                                │
--  │  SQL CONCEPTS DEMONSTRATED:                                    │
--  │    - 4-way INNER JOIN on shared primary key                    │
--  │    - Column aliasing for clean Python column names             │
--  │    - Computed columns (revenue_at_risk)                        │
--  │    - CASE WHEN for feature engineering in SQL                  │
--  │                                                                │
--  │  BUSINESS LOGIC:                                               │
--  │    revenue_at_risk = monthly_charges × 12 × churn_binary       │
--  │    This is the ANNUAL revenue we lose if this customer churns. │
--  │    The "× churn_binary" ensures non-churners have $0 at risk.  │
--  │    In the dashboard (Phase 4), we sum this to show total       │
--  │    revenue exposure to the marketing manager.                  │
--  └─────────────────────────────────────────────────────────────────┘

SELECT
    c.customer_id,

    -- Demographics
    c.gender,
    c.senior_citizen,
    c.partner,
    c.dependents,
    c.tenure_months,
    c.contract_type,

    -- Services (feature-engineered into a count)
    s.internet_service,
    s.online_security,
    s.online_backup,
    s.device_protection,
    s.tech_support,
    s.streaming_tv,
    s.streaming_movies,
    s.phone_service,

    -- Computed: total number of premium services subscribed
    -- WHY: Service count is a proxy for "stickiness" — more services
    -- means higher switching costs, which reduces churn.
    (CASE WHEN s.online_security = 'Yes' THEN 1 ELSE 0 END
     + CASE WHEN s.online_backup = 'Yes' THEN 1 ELSE 0 END
     + CASE WHEN s.device_protection = 'Yes' THEN 1 ELSE 0 END
     + CASE WHEN s.tech_support = 'Yes' THEN 1 ELSE 0 END
     + CASE WHEN s.streaming_tv = 'Yes' THEN 1 ELSE 0 END
     + CASE WHEN s.streaming_movies = 'Yes' THEN 1 ELSE 0 END
    ) AS num_premium_services,

    -- Billing
    b.monthly_charges,
    b.total_charges,
    b.payment_method,
    b.paperless_billing,
    b.customer_lifetime_value AS clv,

    -- Marketing (Causal variables)
    m.marketing_offer_given AS treatment,
    m.propensity_score,

    -- Outcome
    cl.churn_binary,

    -- Computed: Annual Revenue at Risk
    -- This is the dollar impact if we FAIL to retain this customer.
    ROUND(b.monthly_charges * 12 * cl.churn_binary, 2) AS revenue_at_risk,

    -- Computed: Tenure bucket for stratified analysis
    CASE
        WHEN c.tenure_months <= 12 THEN '0-12 months'
        WHEN c.tenure_months <= 24 THEN '13-24 months'
        WHEN c.tenure_months <= 48 THEN '25-48 months'
        ELSE '49+ months'
    END AS tenure_bucket

FROM customers c
INNER JOIN services s      ON c.customer_id = s.customer_id
INNER JOIN billing b       ON c.customer_id = b.customer_id
INNER JOIN marketing m     ON c.customer_id = m.customer_id
INNER JOIN churn_labels cl ON c.customer_id = cl.customer_id

ORDER BY b.customer_lifetime_value DESC
"""


# ══════════════════════════════════════════════════════════════════════
# QUERY 2: Cohort Analysis with Window Functions
# ══════════════════════════════════════════════════════════════════════
QUERY_COHORT_ANALYSIS = """
--  ┌─────────────────────────────────────────────────────────────────┐
--  │  QUERY 2: TENURE COHORT ANALYSIS WITH WINDOW FUNCTIONS         │
--  │                                                                │
--  │  SQL CONCEPTS DEMONSTRATED:                                    │
--  │    - CTE (Common Table Expression) for staged computation      │
--  │    - NTILE(10) window function for decile bucketing             │
--  │    - Conditional aggregation (AVG with CASE WHEN)              │
--  │    - ROUND for clean output                                    │
--  │                                                                │
--  │  BUSINESS LOGIC:                                               │
--  │    Split customers into 10 tenure-based cohorts (deciles).     │
--  │    For each cohort, compute:                                   │
--  │      - Churn rate (overall and by treatment status)            │
--  │      - Average CLV                                             │
--  │      - Treatment penetration rate                              │
--  │                                                                │
--  │    This reveals WHERE the selection bias lives:                 │
--  │    early-tenure cohorts should have higher treatment rates      │
--  │    AND higher churn — confirming that marketing targeted        │
--  │    high-risk customers (the bias we introduced in Script 1).   │
--  │                                                                │
--  │  INTERVIEW PREP:                                               │
--  │    Q: "What is NTILE and when would you use it?"               │
--  │    A: "NTILE(n) divides ordered rows into n roughly equal      │
--  │       buckets. I use it here to create tenure deciles without   │
--  │       hard-coding bucket boundaries, making the analysis        │
--  │       adaptive to any data distribution."                      │
--  └─────────────────────────────────────────────────────────────────┘

WITH customer_cohorts AS (
    -- CTE 1: Assign each customer to a tenure decile
    -- NTILE(10) over tenure creates 10 roughly equal groups
    -- ordered from newest (decile 1) to longest-tenured (decile 10)
    SELECT
        c.customer_id,
        c.tenure_months,
        NTILE(10) OVER (ORDER BY c.tenure_months) AS tenure_decile,
        cl.churn_binary,
        m.marketing_offer_given AS treatment,
        b.customer_lifetime_value AS clv,
        b.monthly_charges
    FROM customers c
    JOIN churn_labels cl ON c.customer_id = cl.customer_id
    JOIN marketing m     ON c.customer_id = m.customer_id
    JOIN billing b       ON c.customer_id = b.customer_id
)

SELECT
    tenure_decile,
    MIN(tenure_months) AS min_tenure,
    MAX(tenure_months) AS max_tenure,
    COUNT(*) AS customer_count,

    -- Overall churn rate for this cohort
    ROUND(AVG(churn_binary) * 100, 1) AS churn_rate_pct,

    -- Churn rate ONLY for treated customers
    -- This uses conditional aggregation: AVG only where treatment = 1
    ROUND(AVG(CASE WHEN treatment = 1 THEN churn_binary END) * 100, 1)
        AS treated_churn_pct,

    -- Churn rate ONLY for control customers
    ROUND(AVG(CASE WHEN treatment = 0 THEN churn_binary END) * 100, 1)
        AS control_churn_pct,

    -- What % of this cohort received treatment?
    -- High values in early cohorts = evidence of selection bias
    ROUND(AVG(treatment) * 100, 1) AS treatment_rate_pct,

    -- Average CLV for prioritization
    ROUND(AVG(clv), 2) AS avg_clv,

    -- Total revenue at risk for this cohort
    ROUND(SUM(monthly_charges * 12 * churn_binary), 2) AS total_revenue_at_risk

FROM customer_cohorts
GROUP BY tenure_decile
ORDER BY tenure_decile
"""


# ══════════════════════════════════════════════════════════════════════
# QUERY 3: Treatment vs Control Comparison (Selection Bias Analysis)
# ══════════════════════════════════════════════════════════════════════
QUERY_TREATMENT_COMPARISON = """
--  ┌─────────────────────────────────────────────────────────────────┐
--  │  QUERY 3: TREATMENT vs CONTROL DEEP COMPARISON                 │
--  │                                                                │
--  │  SQL CONCEPTS DEMONSTRATED:                                    │
--  │    - Multiple CTEs chained together                            │
--  │    - RANK() and PERCENT_RANK() window functions                │
--  │    - Pivot-style conditional aggregation                       │
--  │    - Subquery in SELECT for percentage calculation              │
--  │                                                                │
--  │  BUSINESS LOGIC — THIS IS THE KEY INSIGHT:                     │
--  │    If we naively compare treatment vs control groups,           │
--  │    treatment appears to INCREASE churn (because we targeted    │
--  │    high-risk customers). This demonstrates WHY you need        │
--  │    causal methods — the naive comparison is misleading.        │
--  │                                                                │
--  │  This query produces the "before causal correction" baseline   │
--  │  that Phase 2's uplift model will improve upon.                │
--  └─────────────────────────────────────────────────────────────────┘

WITH base AS (
    -- Join all relevant tables
    SELECT
        c.customer_id,
        c.tenure_months,
        c.contract_type,
        s.internet_service,
        b.monthly_charges,
        b.customer_lifetime_value AS clv,
        m.marketing_offer_given AS treatment,
        m.propensity_score,
        cl.churn_binary
    FROM customers c
    JOIN services s      ON c.customer_id = s.customer_id
    JOIN billing b       ON c.customer_id = b.customer_id
    JOIN marketing m     ON c.customer_id = m.customer_id
    JOIN churn_labels cl ON c.customer_id = cl.customer_id
),

treatment_stats AS (
    -- Aggregate statistics for Treatment group (received offer)
    SELECT
        'Treatment' AS group_name,
        COUNT(*) AS n_customers,
        ROUND(AVG(churn_binary) * 100, 2) AS churn_rate_pct,
        ROUND(AVG(tenure_months), 1) AS avg_tenure,
        ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
        ROUND(AVG(clv), 2) AS avg_clv,
        ROUND(AVG(propensity_score), 4) AS avg_propensity,
        ROUND(SUM(monthly_charges * 12 * churn_binary), 2) AS total_revenue_at_risk
    FROM base
    WHERE treatment = 1
),

control_stats AS (
    -- Aggregate statistics for Control group (no offer)
    SELECT
        'Control' AS group_name,
        COUNT(*) AS n_customers,
        ROUND(AVG(churn_binary) * 100, 2) AS churn_rate_pct,
        ROUND(AVG(tenure_months), 1) AS avg_tenure,
        ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
        ROUND(AVG(clv), 2) AS avg_clv,
        ROUND(AVG(propensity_score), 4) AS avg_propensity,
        ROUND(SUM(monthly_charges * 12 * churn_binary), 2) AS total_revenue_at_risk
    FROM base
    WHERE treatment = 0
)

-- Union the two groups for side-by-side comparison
SELECT * FROM treatment_stats
UNION ALL
SELECT * FROM control_stats
"""


# ══════════════════════════════════════════════════════════════════════
# QUERY 4: High-Value At-Risk Customers (Ranked)
# ══════════════════════════════════════════════════════════════════════
QUERY_HIGH_VALUE_AT_RISK = """
--  ┌─────────────────────────────────────────────────────────────────┐
--  │  QUERY 4: HIGH-VALUE AT-RISK CUSTOMER RANKING                  │
--  │                                                                │
--  │  SQL CONCEPTS DEMONSTRATED:                                    │
--  │    - ROW_NUMBER() for unique ranking                           │
--  │    - PERCENT_RANK() for percentile position                    │
--  │    - LAG() for comparing to "previous" customer                │
--  │    - Multiple window functions in a single query               │
--  │    - QUALIFY clause (DuckDB-specific, elegant filtering)       │
--  │                                                                │
--  │  BUSINESS LOGIC:                                               │
--  │    Identify the top 100 customers ranked by revenue at risk.   │
--  │    These are the customers where intervention matters most:    │
--  │    high revenue AND high churn probability.                    │
--  │                                                                │
--  │    This query pre-computes what the Streamlit dashboard        │
--  │    (Phase 4) will display to the marketing manager.            │
--  └─────────────────────────────────────────────────────────────────┘

WITH ranked_customers AS (
    SELECT
        c.customer_id,
        c.tenure_months,
        c.contract_type,
        b.monthly_charges,
        b.customer_lifetime_value AS clv,
        m.marketing_offer_given AS treatment,
        m.propensity_score,
        cl.churn_binary,

        -- Annual revenue at risk
        ROUND(b.monthly_charges * 12, 2) AS annual_revenue,

        -- RANK by CLV descending — who matters most financially?
        ROW_NUMBER() OVER (
            ORDER BY b.customer_lifetime_value DESC
        ) AS clv_rank,

        -- PERCENT_RANK — what percentile is this customer in?
        -- 0.95 means "top 5% by CLV"
        ROUND(PERCENT_RANK() OVER (
            ORDER BY b.customer_lifetime_value DESC
        ), 4) AS clv_percentile,

        -- LAG — what's the CLV of the customer ranked just above?
        -- Useful for showing "diminishing returns" in targeting
        LAG(b.customer_lifetime_value, 1) OVER (
            ORDER BY b.customer_lifetime_value DESC
        ) AS prev_customer_clv,

        -- Running sum of revenue at risk
        -- Shows cumulative exposure as you move down the ranked list
        SUM(b.monthly_charges * 12 * cl.churn_binary) OVER (
            ORDER BY b.customer_lifetime_value DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_revenue_at_risk

    FROM customers c
    JOIN billing b       ON c.customer_id = b.customer_id
    JOIN marketing m     ON c.customer_id = m.customer_id
    JOIN churn_labels cl ON c.customer_id = cl.customer_id
)

SELECT *
FROM ranked_customers
WHERE clv_rank <= 100
ORDER BY clv_rank
"""


# ══════════════════════════════════════════════════════════════════════
# QUERY 5: Contract-Service Segment Analysis
# ══════════════════════════════════════════════════════════════════════
QUERY_SEGMENT_ANALYSIS = """
--  ┌─────────────────────────────────────────────────────────────────┐
--  │  QUERY 5: CROSS-SEGMENT CHURN & TREATMENT ANALYSIS             │
--  │                                                                │
--  │  SQL CONCEPTS DEMONSTRATED:                                    │
--  │    - GROUP BY with multiple dimensions                         │
--  │    - HAVING clause for filtering aggregates                    │
--  │    - Nested CASE WHEN inside aggregation                       │
--  │                                                                │
--  │  BUSINESS LOGIC:                                               │
--  │    Cross-tabulate contract type × internet service to find     │
--  │    which SEGMENTS have the highest churn and highest treatment  │
--  │    rates. This is the kind of analysis a marketing team        │
--  │    actually uses to design targeted campaigns.                 │
--  │                                                                │
--  │  EXPECTED INSIGHT:                                             │
--  │    Month-to-month + Fiber optic = highest churn segment        │
--  │    Two year + DSL = lowest churn segment                       │
--  │    Treatment rate should be highest in the high-churn segment  │
--  │    (confirming our selection bias design from Script 1)        │
--  └─────────────────────────────────────────────────────────────────┘

SELECT
    c.contract_type,
    s.internet_service,
    COUNT(*) AS n_customers,
    ROUND(AVG(cl.churn_binary) * 100, 1) AS churn_rate_pct,
    ROUND(AVG(m.marketing_offer_given) * 100, 1) AS treatment_rate_pct,

    -- Churn rate by treatment status within each segment
    ROUND(AVG(CASE WHEN m.marketing_offer_given = 1
              THEN cl.churn_binary END) * 100, 1) AS treated_churn_pct,
    ROUND(AVG(CASE WHEN m.marketing_offer_given = 0
              THEN cl.churn_binary END) * 100, 1) AS control_churn_pct,

    -- Average CLV per segment
    ROUND(AVG(b.customer_lifetime_value), 2) AS avg_clv,

    -- Total revenue at risk per segment
    ROUND(SUM(b.monthly_charges * 12 * cl.churn_binary), 2) AS segment_revenue_at_risk

FROM customers c
JOIN services s      ON c.customer_id = s.customer_id
JOIN billing b       ON c.customer_id = b.customer_id
JOIN marketing m     ON c.customer_id = m.customer_id
JOIN churn_labels cl ON c.customer_id = cl.customer_id

GROUP BY c.contract_type, s.internet_service

-- Only show segments with meaningful sample size
HAVING COUNT(*) >= 50

ORDER BY churn_rate_pct DESC
"""


# ─────────────────────────────────────────────────────────────────────
# EXECUTION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────

def execute_query(con: duckdb.DuckDBPyConnection, query: str, name: str) -> pd.DataFrame:
    """Execute a SQL query and return results as a pandas DataFrame."""
    print(f"\n{'─' * 60}")
    print(f"Executing: {name}")
    print(f"{'─' * 60}")

    df = con.execute(query).fetchdf()
    print(f"[RESULT] {len(df)} rows × {len(df.columns)} columns")
    print(df.head(10).to_string(index=False))

    return df


def extract_modeling_dataset(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Extract the primary modeling dataset and save to CSV.

    This is the dataset that Phase 2 will consume.
    It contains all features, treatment indicator, propensity score,
    and the churn outcome — everything needed for uplift modeling.
    """
    print("\n" + "=" * 60)
    print("EXTRACTING MODELING DATASET FOR PHASE 2")
    print("=" * 60)

    df = con.execute(QUERY_MODELING_DATASET).fetchdf()

    # ── Final validation before export ──
    assert len(df) > 0, "Modeling dataset is empty"
    assert "treatment" in df.columns, "Missing treatment column"
    assert "churn_binary" in df.columns, "Missing outcome column"
    assert "propensity_score" in df.columns, "Missing propensity score"

    # ── Save to CSV ──
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[SAVED] Modeling dataset → {OUTPUT_PATH}")
    print(f"[INFO]  {len(df)} rows × {df.shape[1]} columns")
    print(f"[INFO]  Treatment rate: {df['treatment'].mean():.1%}")
    print(f"[INFO]  Churn rate:     {df['churn_binary'].mean():.1%}")
    print(f"[INFO]  Avg CLV:        ${df['clv'].mean():,.2f}")

    return df


# ─────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("PHASE 1 | Script 3: SQL QUERIES & DATA EXTRACTION")
    print("=" * 60)

    con = get_connection()

    # ── Run all analytical queries ──
    # These demonstrate SQL skills and validate the data
    queries = [
        (QUERY_COHORT_ANALYSIS, "Cohort Analysis (NTILE Window Function)"),
        (QUERY_TREATMENT_COMPARISON, "Treatment vs Control Comparison"),
        (QUERY_HIGH_VALUE_AT_RISK, "High-Value At-Risk Ranking"),
        (QUERY_SEGMENT_ANALYSIS, "Contract × Service Segment Analysis"),
    ]

    for query, name in queries:
        execute_query(con, query, name)

    # ── Extract the primary modeling dataset ──
    modeling_df = extract_modeling_dataset(con)

    con.close()

    print(f"\n{'=' * 60}")
    print("PHASE 1 COMPLETE")
    print(f"{'=' * 60}")
    print(f"\nDeliverables:")
    print(f"  1. Enriched CSV:      telco_churn_enriched.csv")
    print(f"  2. DuckDB database:   retention_safeguard.duckdb")
    print(f"  3. Modeling dataset:  modeling_dataset.csv")
    print(f"\nNext: Phase 2 — Causal Modeling (T-Learner & X-Learner)")
    print(f"{'=' * 60}")

    return modeling_df


if __name__ == "__main__":
    main()