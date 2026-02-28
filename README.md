# 🛡️ Proactive Retention & Revenue Safeguard System

**A Causal AI application that identifies *who you can save*, not just who will leave.**

> Traditional churn models predict who will leave. This system predicts who you can **change** — using uplift modeling to maximize retention ROI.

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?style=flat-square&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=flat-square&logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=flat-square&logo=docker)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 📌 The Problem

Every company has a churn model. Most of them answer the wrong question.

Predicting that a customer has a 90% chance of churning is useless if **nothing you do will change their mind** (Lost Cause). Meanwhile, a customer with only 40% churn risk might be the exact person a $50 retention offer converts into a loyal, high-value customer (Persuadable).

**Standard churn models waste marketing budget on customers who would have stayed anyway (Sure Things) or who will leave no matter what (Lost Causes). Worse, they might target customers who churn *because* of the outreach (Sleeping Dogs).**

This project solves that problem using **Causal AI** — specifically, uplift modeling via T-Learner and X-Learner architectures.

---

## 🎯 What This System Does

For every customer, the system answers four questions:

| Question | How It's Answered |
|----------|-------------------|
| Will they churn? | Churn probability from gradient boosting |
| Can we change that? | Uplift score (CATE) from T-Learner |
| Which segment are they? | Four-quadrant classification |
| How much is at stake? | Dollar value-at-risk calculation |

### The Four Uplift Quadrants

| Quadrant | What It Means | Action |
|----------|---------------|--------|
| 🎯 **Persuadable** | Would churn WITHOUT offer, stays WITH offer | **Target aggressively — highest ROI** |
| ✅ **Sure Thing** | Stays regardless of offer | Don't waste budget |
| ❌ **Lost Cause** | Churns regardless of offer | Deprioritize |
| 🛑 **Sleeping Dog** | Would stay, but churns BECAUSE of offer | **Never contact** |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    FULL-STACK PIPELINE                        │
├──────────────┬──────────────┬──────────────┬─────────────────┤
│   Phase 1    │   Phase 2    │   Phase 3    │    Phase 4      │
│   Data Eng   │   Modeling   │   API        │    Dashboard    │
├──────────────┼──────────────┼──────────────┼─────────────────┤
│ IBM Telco    │ T-Learner    │ FastAPI      │ Streamlit       │
│ Dataset      │ X-Learner    │ /predict     │ Budget Optimizer│
│ DuckDB       │ IPW          │ /batch       │ Quadrant View   │
│ SQL Pipeline │ Qini Curves  │ /health      │ CSV Download    │
│ Propensity   │ Fairness     │ Pydantic v2  │ Live Scorer     │
│ Score Gen    │ Analysis     │ Docker       │                 │
└──────────────┴──────────────┴──────────────┴─────────────────┘
```

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)
```bash
git clone https://github.com/UtsavKhadka-Analyst/retention-safeguard.git
cd retention-safeguard
docker-compose up
```
Then visit:
- **Dashboard:** http://localhost:8501
- **API Docs:** http://localhost:8000/docs

### Option 2: Manual Setup
```bash
git clone https://github.com/UtsavKhadka-Analyst/retention-safeguard.git
cd retention-safeguard
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Run the pipeline:
```bash
# Phase 1: Generate data and load to database
python phase1_data/fetch_and_enrich_data.py
python phase1_data/load_to_db.py
python phase1_data/sql_queries.py

# Phase 2: Train causal models
python phase2_modeling/t_learner.py
python phase2_modeling/x_learner.py
python phase2_modeling/propensity.py
python phase2_modeling/evaluation.py
python phase2_modeling/fairness.py

# Phase 3: Start API (Terminal 1)
cd phase3_api && uvicorn main:app --port 8000

# Phase 4: Start Dashboard (Terminal 2)
cd phase4_dashboard && streamlit run app.py
```

---

## 📊 Key Results

### Model Performance

| Metric | T-Learner | X-Learner |
|--------|-----------|-----------|
| Qini Coefficient | **0.115** | 0.064 |
| Uplift@10% | **0.60** | 0.20 |
| Uplift@20% | **0.48** | 0.15 |
| Score Correlation | 0.77 | 0.77 |

**T-Learner outperformed** on this dataset, capturing 3x more treatment effect in the top decile.

### Business Impact

| Metric | Value |
|--------|-------|
| Total Customers | 7,043 |
| Persuadable Customers | ~2,200 (31%) |
| Saveable Revenue | ~$324,000 |
| Sleeping Dogs Identified | ~3,800 (54%) |
| Value Protected (by NOT contacting) | ~$719,000 |

### Selection Bias Correction (IPW)

| Estimate | Naive (Biased) | IPW (Corrected) |
|----------|---------------|-----------------|
| Treatment Effect on Churn | -16.3pp | -3.8pp |
| Interpretation | Offer "increases" churn | Bias mostly corrected |

The naive estimate wrongly suggests the offer increases churn by 16.3 percentage points. After IPW correction, the bias is reduced by **76%** — demonstrating why causal methods are necessary.

### Fairness Analysis

| Demographic | Parity Ratio | Status |
|-------------|-------------|--------|
| Gender | 0.992 | ✅ Pass |
| Senior Citizen | 0.946 | ✅ Pass |
| Partner Status | 0.868 | ✅ Pass |

All demographic groups pass the 80% parity threshold — no group is disproportionately classified as Persuadable or Sleeping Dog.

---

## 🧮 Mathematical Framework

### Potential Outcomes (Rubin Causal Model)

For each customer *i*, two potential outcomes exist:
- **Y(1)**: outcome if treated (given retention offer)
- **Y(0)**: outcome if not treated

The **Individual Treatment Effect**: ITE(i) = Y(1) - Y(0)

The fundamental problem: we only observe ONE outcome per customer.

### Conditional Average Treatment Effect (CATE)

Since we can't observe individual effects, we estimate:

**τ(X) = E[Y(0) | X] - E[Y(1) | X]**

Positive τ means the offer **reduces** churn (since Y=1 is churn).

### T-Learner Architecture

Two separate models trained on subgroups:
- **μ₀(x)** = E[Y | X=x, T=0] — trained on control group
- **μ₁(x)** = E[Y | X=x, T=1] — trained on treatment group
- **Uplift** = μ₀(x) - μ₁(x)

### X-Learner (3-Stage)

Improves on T-Learner with cross-imputation:
1. Train outcome models (same as T-Learner)
2. Impute treatment effects: D¹ = Y(1) - μ₀(X), D⁰ = μ₁(X) - Y(0)
3. Train effect models on imputed values
4. Combine using propensity-weighted average: τ(x) = e(x)·τ₀(x) + (1-e(x))·τ₁(x)

### Inverse Probability Weighting (IPW)

Corrects selection bias by reweighting observations:
- Treated: weight = 1 / e(x)
- Control: weight = 1 / (1 - e(x))

Where e(x) = P(Treatment=1 | X) is the propensity score.

---

## 📁 Project Structure

```
retention-safeguard/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
│
├── phase1_data/                  # Data Engineering
│   ├── fetch_and_enrich_data.py  # Download + synthetic enrichment
│   ├── load_to_db.py             # DuckDB normalized schema
│   └── sql_queries.py            # Complex JOINs, window functions
│
├── phase2_modeling/              # Causal Inference
│   ├── t_learner.py              # T-Learner (baseline)
│   ├── x_learner.py              # X-Learner (advanced)
│   ├── propensity.py             # IPW bias correction
│   ├── evaluation.py             # Qini curves, uplift@k
│   ├── fairness.py               # Demographic parity analysis
│   └── artifacts/                # Saved models and scored data
│
├── phase3_api/                   # Production API
│   ├── main.py                   # FastAPI endpoints
│   ├── schemas.py                # Pydantic request/response models
│   └── model_loader.py           # Model loading and inference
│
└── phase4_dashboard/             # Stakeholder UI
    └── app.py                    # Streamlit marketing dashboard
```

---

## ⚠️ Honest Limitations

**I believe transparency about limitations is a strength, not a weakness.**

### 1. Synthetic Treatment Data
The `Marketing_Offer_Given` column is synthetically generated. No public dataset contains real experimental retention data with randomized treatment assignment — this is always proprietary. The causal pipeline is production-ready; swap in real experimental data and it works identically.

### 2. Selection Bias by Design
Treatment was assigned non-randomly (high-risk customers were more likely to receive offers) to simulate real-world conditions. IPW partially corrects this, but hidden confounders — variables that affect both treatment and outcome but aren't in our dataset — cannot be addressed without a true randomized experiment.

### 3. SUTVA Assumption
The Stable Unit Treatment Value Assumption requires that one customer's treatment doesn't affect another's outcome. Network effects (e.g., family plans) would violate this in production.

### 4. What I'd Do With Real Data
- Run a **randomized controlled trial** (A/B test) for unbiased treatment effects
- Apply **doubly robust estimators** (combining IPW and outcome modeling)
- Use **sensitivity analysis** (Rosenbaum bounds) to assess hidden confounding
- Implement **model drift detection** with production monitoring

---

## 🎓 Skills Demonstrated

| Category | Skills |
|----------|--------|
| **Data Engineering** | SQL (JOINs, CTEs, window functions), DuckDB, normalized schemas |
| **Causal Inference** | T-Learner, X-Learner, CATE estimation, potential outcomes framework |
| **Bias Correction** | Propensity scores, IPW, covariate balance diagnostics |
| **ML Engineering** | Gradient boosting, cross-validation, model serialization |
| **Evaluation** | Qini curves, uplift@k, cumulative gain (not just AUC) |
| **Fairness** | Demographic parity, equalized uplift, proportional representation |
| **Software Engineering** | FastAPI, Pydantic v2, Docker, modular architecture |
| **Product Thinking** | ROI-driven dashboard, budget optimization, actionable recommendations |

---

## 🧑‍💼 About

**Utsav Khadka** — MS in Analytics (graduating May 2026)

This project was built as a flagship portfolio piece to demonstrate causal inference, production ML engineering, and business-oriented data science. Every design decision — from choosing uplift modeling over basic classification to including a fairness analysis — is calibrated to signal depth and maturity.

- GitHub: [UtsavKhadka-Analyst](https://github.com/UtsavKhadka-Analyst)
- Target roles: Data Scientist | Senior Data Analyst | ML Engineer

---

## 📝 License

MIT License — feel free to use this as a reference for your own projects.