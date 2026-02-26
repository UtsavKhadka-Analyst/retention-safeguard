"""
=============================================================================
PHASE 1 | Script 2 of 3: load_to_db.py
=============================================================================
Proactive Retention & Revenue Safeguard System
Author: Utsav Khadka (github.com/UtsavKhadka-Analyst)

PURPOSE:
    Load the enriched CSV from Script 1 into a local DuckDB database.
    Create a normalized schema with multiple tables to enable
    complex JOINs in Script 3.

WHY DUCKDB?
    - Embedded analytical database (no server setup, like SQLite but faster)
    - Column-oriented storage — optimized for analytics workloads
    - Native pandas integration — pd.read_sql() works seamlessly
    - Used increasingly in industry (dbt, Motherduck, analytics engineering)
    - Shows you know modern data tooling, not just "I can use SQLite"

WHY MULTIPLE TABLES?
    A single flat CSV is fine for quick analysis. But in production,
    data lives in normalized tables. Creating JOINable tables demonstrates:
    1. You understand relational modeling
    2. You can write non-trivial SQL (JOINs, not just SELECT *)
    3. You think about data engineering, not just data science

SCHEMA DESIGN:
    customers        → Demographics and account info
    services         → What services each customer subscribes to
    billing          → Financial data (charges, payment, CLV)
    marketing        → Treatment assignment and propensity scores
    churn_labels     → Outcome variable (kept separate for clean modeling)

INTERVIEW PREP:
    Q: "Why did you normalize the data into multiple tables?"
    A: "In production, customer data lives across multiple systems —
       CRM, billing, marketing automation. My SQL queries demonstrate
       I can work with data as it actually exists in companies,
       not just pre-joined flat files."
=============================================================================
"""

import duckdb
import pandas as pd
import os
import sys

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "telco_churn_enriched.csv")
DB_PATH = os.path.join(SCRIPT_DIR, "retention_safeguard.duckdb")


def validate_input_data(csv_path: str) -> pd.DataFrame:
    """
    Load and validate the enriched CSV before database insertion.

    DATA QUALITY CHECKS:
        These are the kinds of checks a data engineer runs in production
        (often via tools like Great Expectations or dbt tests).
        They catch issues BEFORE they corrupt your database.
    """
    print("[INFO] Validating input data...")

    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        print(f"[HINT]  Run fetch_and_enrich_data.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # ── Required columns check ──
    required_cols = [
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges",
        "Churn", "Churn_Binary",
        "Marketing_Offer_Given", "Customer_Lifetime_Value", "Propensity_Score"
    ]

    missing = set(required_cols) - set(df.columns)
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        sys.exit(1)

    # ── Data type and range checks ──
    assert df["customerID"].nunique() == len(df), "Duplicate customerIDs found"
    assert df["Churn_Binary"].isin([0, 1]).all(), "Churn_Binary must be 0 or 1"
    assert df["Marketing_Offer_Given"].isin([0, 1]).all(), "Treatment must be 0 or 1"
    assert df["Propensity_Score"].between(0, 1).all(), "Propensity scores out of range"
    assert df["Customer_Lifetime_Value"].min() > 0, "CLV must be positive"

    print(f"[SUCCESS] Validation passed: {len(df)} rows, {len(df.columns)} columns")
    return df


def create_database(db_path: str, df: pd.DataFrame) -> None:
    """
    Create a normalized DuckDB database from the flat enriched CSV.

    NORMALIZATION STRATEGY:
        We split the single DataFrame into 5 related tables:

        customers ──┬── services     (1:1 on customerID)
                    ├── billing      (1:1 on customerID)
                    ├── marketing    (1:1 on customerID)
                    └── churn_labels (1:1 on customerID)

        In a real company, these would be separate source systems:
        - customers:    CRM (Salesforce, HubSpot)
        - services:     Product database
        - billing:      Financial system (Stripe, Zuora)
        - marketing:    Marketing automation (Marketo, Braze)
        - churn_labels: Analytics team's labeled outcomes

    WHY 1:1 RELATIONSHIPS?
        The original data is customer-level (one row per customer).
        In reality, billing might be 1:many (monthly invoices).
        We keep 1:1 for simplicity but the JOINs demonstrate the
        same SQL skills needed for more complex schemas.
    """
    print(f"\n[INFO] Creating DuckDB database at: {db_path}")

    # ── Remove existing database for clean rebuild ──
    if os.path.exists(db_path):
        os.remove(db_path)
        print("[INFO] Removed existing database.")

    # ── Connect to DuckDB ──
    con = duckdb.connect(db_path)

    # ══════════════════════════════════════════════════════════════════
    # TABLE 1: customers — Demographics and account information
    # ══════════════════════════════════════════════════════════════════
    con.execute("""
        CREATE TABLE customers (
            customer_id     VARCHAR PRIMARY KEY,
            gender          VARCHAR NOT NULL,
            senior_citizen  INTEGER NOT NULL,
            partner         VARCHAR NOT NULL,
            dependents      VARCHAR NOT NULL,
            tenure_months   INTEGER NOT NULL,
            contract_type   VARCHAR NOT NULL,

            -- COMMENTARY:
            -- tenure_months is the single most important feature for churn.
            -- Customers with < 12 months tenure churn at 3-4x the rate of
            -- customers with 48+ months. This is consistent across industries.
            -- contract_type is the second strongest predictor.

            CONSTRAINT chk_tenure CHECK (tenure_months >= 0),
            CONSTRAINT chk_senior CHECK (senior_citizen IN (0, 1))
        )
    """)
    print("[CREATED] Table: customers")

    # ══════════════════════════════════════════════════════════════════
    # TABLE 2: services — Subscribed products and features
    # ══════════════════════════════════════════════════════════════════
    con.execute("""
        CREATE TABLE services (
            customer_id       VARCHAR PRIMARY KEY,
            phone_service     VARCHAR NOT NULL,
            multiple_lines    VARCHAR NOT NULL,
            internet_service  VARCHAR NOT NULL,
            online_security   VARCHAR NOT NULL,
            online_backup     VARCHAR NOT NULL,
            device_protection VARCHAR NOT NULL,
            tech_support      VARCHAR NOT NULL,
            streaming_tv      VARCHAR NOT NULL,
            streaming_movies  VARCHAR NOT NULL,

            -- COMMENTARY:
            -- Service subscriptions serve dual purpose:
            -- 1. Feature input for the churn/uplift model
            -- 2. Proxy for customer engagement and stickiness
            -- Customers with more services have higher switching costs
            -- → lower churn. This is the "bundle retention" strategy
            -- that telecom companies actively exploit.

            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        )
    """)
    print("[CREATED] Table: services")

    # ══════════════════════════════════════════════════════════════════
    # TABLE 3: billing — Financial data
    # ══════════════════════════════════════════════════════════════════
    con.execute("""
        CREATE TABLE billing (
            customer_id             VARCHAR PRIMARY KEY,
            monthly_charges         DOUBLE NOT NULL,
            total_charges           DOUBLE NOT NULL,
            payment_method          VARCHAR NOT NULL,
            paperless_billing       VARCHAR NOT NULL,
            customer_lifetime_value DOUBLE NOT NULL,

            -- COMMENTARY:
            -- CLV is our enriched column from Script 1.
            -- In a real company, CLV would come from the finance team's model.
            -- monthly_charges × expected_remaining_lifetime = revenue at risk
            -- This converts a statistical prediction into a dollar figure
            -- that a non-technical stakeholder can act on.

            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            CONSTRAINT chk_monthly CHECK (monthly_charges >= 0),
            CONSTRAINT chk_clv CHECK (customer_lifetime_value > 0)
        )
    """)
    print("[CREATED] Table: billing")

    # ══════════════════════════════════════════════════════════════════
    # TABLE 4: marketing — Treatment assignment and propensity
    # ══════════════════════════════════════════════════════════════════
    con.execute("""
        CREATE TABLE marketing (
            customer_id          VARCHAR PRIMARY KEY,
            marketing_offer_given INTEGER NOT NULL,
            propensity_score     DOUBLE NOT NULL,

            -- COMMENTARY:
            -- This table is the CAUSAL core of the project.
            -- marketing_offer_given = Treatment indicator (T ∈ {0, 1})
            -- propensity_score = P(T=1 | X) — estimated by logistic regression
            --
            -- In a real company, this data comes from the marketing automation
            -- platform (who was sent an email, who got a phone call, etc.)
            -- The propensity score would be computed by the data science team
            -- to correct for the non-random targeting decisions.

            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            CONSTRAINT chk_treatment CHECK (marketing_offer_given IN (0, 1)),
            CONSTRAINT chk_propensity CHECK (propensity_score BETWEEN 0 AND 1)
        )
    """)
    print("[CREATED] Table: marketing")

    # ══════════════════════════════════════════════════════════════════
    # TABLE 5: churn_labels — Outcome variable
    # ══════════════════════════════════════════════════════════════════
    con.execute("""
        CREATE TABLE churn_labels (
            customer_id  VARCHAR PRIMARY KEY,
            churn_label  VARCHAR NOT NULL,
            churn_binary INTEGER NOT NULL,

            -- COMMENTARY:
            -- The outcome is separated because in production:
            -- 1. Labels might arrive days/weeks after features (delayed feedback)
            -- 2. You might retrain models on new labels without touching feature tables
            -- 3. It enforces clean separation of X (features) and Y (target)

            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            CONSTRAINT chk_churn CHECK (churn_binary IN (0, 1))
        )
    """)
    print("[CREATED] Table: churn_labels")

    # ══════════════════════════════════════════════════════════════════
    # INSERT DATA INTO TABLES
    # ══════════════════════════════════════════════════════════════════
    print("\n[INFO] Inserting data into tables...")

    # ── Table: customers ──
    customers_df = df[[
        "customerID", "gender", "SeniorCitizen", "Partner",
        "Dependents", "tenure", "Contract"
    ]].copy()
    customers_df.columns = [
        "customer_id", "gender", "senior_citizen", "partner",
        "dependents", "tenure_months", "contract_type"
    ]
    con.execute("INSERT INTO customers SELECT * FROM customers_df")
    print(f"  → customers:    {len(customers_df):,} rows")

    # ── Table: services ──
    services_df = df[[
        "customerID", "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]].copy()
    services_df.columns = [
        "customer_id", "phone_service", "multiple_lines", "internet_service",
        "online_security", "online_backup", "device_protection", "tech_support",
        "streaming_tv", "streaming_movies"
    ]
    con.execute("INSERT INTO services SELECT * FROM services_df")
    print(f"  → services:     {len(services_df):,} rows")

    # ── Table: billing ──
    billing_df = df[[
        "customerID", "MonthlyCharges", "TotalCharges",
        "PaymentMethod", "PaperlessBilling", "Customer_Lifetime_Value"
    ]].copy()
    billing_df.columns = [
        "customer_id", "monthly_charges", "total_charges",
        "payment_method", "paperless_billing", "customer_lifetime_value"
    ]
    con.execute("INSERT INTO billing SELECT * FROM billing_df")
    print(f"  → billing:      {len(billing_df):,} rows")

    # ── Table: marketing ──
    marketing_df = df[[
        "customerID", "Marketing_Offer_Given", "Propensity_Score"
    ]].copy()
    marketing_df.columns = [
        "customer_id", "marketing_offer_given", "propensity_score"
    ]
    con.execute("INSERT INTO marketing SELECT * FROM marketing_df")
    print(f"  → marketing:    {len(marketing_df):,} rows")

    # ── Table: churn_labels ──
    churn_df = df[["customerID", "Churn", "Churn_Binary"]].copy()
    churn_df.columns = ["customer_id", "churn_label", "churn_binary"]
    con.execute("INSERT INTO churn_labels SELECT * FROM churn_df")
    print(f"  → churn_labels: {len(churn_df):,} rows")

    # ══════════════════════════════════════════════════════════════════
    # VERIFY: Count rows in each table
    # ══════════════════════════════════════════════════════════════════
    print("\n[INFO] Verifying table integrity...")
    tables = ["customers", "services", "billing", "marketing", "churn_labels"]
    for table in tables:
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  ✓ {table}: {count:,} rows")

    # ── Verify referential integrity with a quick JOIN ──
    join_count = con.execute("""
        SELECT COUNT(*)
        FROM customers c
        JOIN services s ON c.customer_id = s.customer_id
        JOIN billing b ON c.customer_id = b.customer_id
        JOIN marketing m ON c.customer_id = m.customer_id
        JOIN churn_labels cl ON c.customer_id = cl.customer_id
    """).fetchone()[0]
    print(f"  ✓ Full JOIN:  {join_count:,} rows (should equal customer count)")

    assert join_count == len(df), "JOIN count mismatch — referential integrity broken!"

    con.close()
    print(f"\n[SUCCESS] Database created: {db_path}")
    print(f"[INFO]    5 tables, {len(df):,} rows each, all foreign keys valid")


def print_schema_summary(db_path: str) -> None:
    """
    Print a nice summary of the database schema.
    Useful for documentation and README screenshots.
    """
    con = duckdb.connect(db_path, read_only=True)

    print("\n" + "=" * 60)
    print("DATABASE SCHEMA SUMMARY")
    print("=" * 60)

    tables = con.execute("SHOW TABLES").fetchall()
    for (table_name,) in tables:
        print(f"\n┌── {table_name} ──")
        columns = con.execute(f"DESCRIBE {table_name}").fetchall()
        for col in columns:
            col_name, col_type = col[0], col[1]
            print(f"│   {col_name:<30} {col_type}")
        print(f"└── {con.execute(f'SELECT COUNT(*) FROM {table_name}').fetchone()[0]:,} rows")

    con.close()


# ─────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("PHASE 1 | Script 2: DATABASE LOADER")
    print("=" * 60 + "\n")

    # Step 1: Validate input CSV
    df = validate_input_data(CSV_PATH)

    # Step 2: Create and populate database
    create_database(DB_PATH, df)

    # Step 3: Print schema summary
    print_schema_summary(DB_PATH)

    print(f"\n{'=' * 60}")
    print("Phase 1, Script 2 COMPLETE. Proceed to Script 3 (sql_queries.py)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()