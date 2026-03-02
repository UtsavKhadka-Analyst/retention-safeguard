"""
=============================================================================
PHASE 3 | Script 3 of 3: main.py
=============================================================================
Proactive Retention & Revenue Safeguard System
Author: Utsav Khadka (github.com/UtsavKhadka-Analyst)

PURPOSE:
    FastAPI application that serves the T-Learner uplift model as a
    REST API with three endpoints:
        /predict  → Score a single customer
        /batch    → Score multiple customers
        /health   → Model status and metadata

WHY FASTAPI?
    1. Automatic OpenAPI/Swagger documentation (visit /docs)
    2. Built-in request validation via Pydantic
    3. Async support for high-throughput serving
    4. Type hints everywhere — self-documenting code
    5. Industry standard for ML model serving in Python

HOW TO RUN:
    uvicorn main:app --reload --port 8000

    Then visit:
        http://localhost:8000/docs  → Interactive API documentation
        http://localhost:8000/health → Model health check

PRODUCTION CONSIDERATIONS (for interviews):
    - In production, you'd add: authentication (JWT/API keys),
      rate limiting, request logging, model versioning (A/B testing),
      and monitoring (Prometheus + Grafana).
    - For higher throughput: use gunicorn with uvicorn workers,
      or deploy behind a load balancer.
    - For model updates: implement blue-green deployment so you can
      swap models without downtime.

=============================================================================
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import time

# ── Import our modules ──
from schemas import (
    CustomerInput,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
)
from model_loader import ModelLoader


# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────

# Model artifacts directory (from Phase 2)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "phase2_modeling", "artifacts")

# ── Initialize model loader (singleton pattern) ──
loader = ModelLoader(model_dir=MODEL_DIR)


# ─────────────────────────────────────────────────────────────────────
# APP LIFESPAN — Load models at startup
# ─────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load models when the API starts, clean up when it stops.

    LIFESPAN PATTERN:
        FastAPI's recommended way to handle startup/shutdown logic.
        Models are loaded ONCE into memory, shared across all requests.
        This avoids the latency of loading models per-request.

    WHY NOT @app.on_event("startup")?
        That decorator is deprecated in FastAPI. Lifespan is the
        modern replacement that properly handles cleanup.
    """
    # ── Startup ──
    print("=" * 50)
    print("  STARTING RETENTION SAFEGUARD API")
    print("=" * 50)

    try:
        loader.load()
        print("[API] Models loaded successfully.")
        print(f"[API] Model directory: {MODEL_DIR}")
    except FileNotFoundError as e:
        print(f"[API] WARNING: {e}")
        print("[API] API will start but /predict will fail.")
        print("[API] Run Phase 2 scripts first to generate model artifacts.")

    print("=" * 50)
    yield
    # ── Shutdown ──
    print("[API] Shutting down...")


# ─────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Retention Safeguard API",
    description=(
        "Causal AI-powered customer retention system. "
        "Uses T-Learner uplift modeling to identify Persuadable customers "
        "and maximize retention ROI. Built by Utsav Khadka."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS middleware ──
# Allows the Streamlit frontend (Phase 4) to call this API.
# In production, restrict origins to your specific frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    """API root — basic info and links."""
    return {
        "name": "Retention Safeguard API",
        "version": "1.0.0",
        "author": "Utsav Khadka",
        "docs": "/docs",
        "health": "/health",
        "endpoints": ["/predict", "/batch", "/health"],
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.

    PRODUCTION USE:
        Load balancers and orchestrators (Kubernetes) call this endpoint
        to verify the service is running and the model is loaded.
        If this returns non-200, the container gets restarted.

    WHAT IT RETURNS:
        - API status (healthy/degraded)
        - Model type and version
        - Training metadata
    """
    if not loader.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run Phase 2 first."
        )

    return HealthResponse(
        status="healthy",
        model_type=loader.model_metadata.get("model_type", "unknown"),
        n_training_samples=loader.model_metadata.get("n_training_samples", 0),
        training_date=loader.model_metadata.get("training_date", "unknown"),
        n_features=loader.model_metadata.get("n_features", 0),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_single(customer: CustomerInput):
    """
    Score a single customer for churn risk, uplift, and quadrant.

    BUSINESS WORKFLOW:
        1. CRM system sends customer data to this endpoint
        2. API runs T-Learner inference (two models, one call)
        3. Returns: churn risk, uplift score, quadrant, recommendation
        4. Marketing team acts on the recommendation

    WHAT MAKES THIS ENDPOINT SPECIAL:
        It doesn't just return a churn probability (every basic model does that).
        It returns the CAUSAL TREATMENT EFFECT — whether an intervention
        would actually help this specific customer. That's the difference
        between prediction and prescriptive analytics.

    EXAMPLE REQUEST:
        POST /predict
        {
            "gender": "Female",
            "senior_citizen": 0,
            "partner": "Yes",
            "dependents": "No",
            "tenure_months": 5,
            "contract_type": "Month-to-month",
            "internet_service": "Fiber optic",
            "online_security": "No",
            ...
        }

    EXAMPLE RESPONSE:
        {
            "churn_risk": 0.72,
            "uplift_score": 0.25,
            "quadrant": "Persuadable",
            "value_at_risk": 487.50,
            "recommendation": "HIGH PRIORITY: Target with retention offer..."
        }
    """
    if not loader.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is starting up."
        )

    try:
        # ── Convert Pydantic model to dict for the loader ──
        # .value extracts the string from Enum types
        customer_dict = {
            "gender": customer.gender,
            "senior_citizen": customer.senior_citizen,
            "partner": customer.partner.value,
            "dependents": customer.dependents.value,
            "tenure_months": customer.tenure_months,
            "contract_type": customer.contract_type.value,
            "internet_service": customer.internet_service.value,
            "online_security": customer.online_security.value,
            "online_backup": customer.online_backup.value,
            "device_protection": customer.device_protection.value,
            "tech_support": customer.tech_support.value,
            "streaming_tv": customer.streaming_tv.value,
            "streaming_movies": customer.streaming_movies.value,
            "phone_service": customer.phone_service.value,
            "multiple_lines": customer.multiple_lines.value,
            "monthly_charges": customer.monthly_charges,
            "total_charges": customer.total_charges,
            "payment_method": customer.payment_method.value,
            "paperless_billing": customer.paperless_billing.value,
        }

        # ── Run prediction ──
        start_time = time.time()
        result = loader.predict(customer_dict)
        latency_ms = (time.time() - start_time) * 1000

        print(f"[PREDICT] Quadrant: {result['quadrant']}, "
              f"Uplift: {result['uplift_score']:+.4f}, "
              f"Latency: {latency_ms:.1f}ms")

        return PredictionResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(customers: list[CustomerInput]):
    """
    Score multiple customers in a single request.

    BUSINESS USE CASE:
        Marketing team uploads their entire customer base (or a segment)
        to get uplift scores for everyone. The response includes:
        - Individual predictions for each customer
        - Aggregate metrics: total revenue at risk, Persuadable count

    PRODUCTION NOTE:
        For very large batches (>10K customers), you'd use an async
        job queue (Celery/Redis) instead of a synchronous endpoint.
        This endpoint is designed for batches up to ~1000 customers.
    """
    if not loader.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded."
        )

    if len(customers) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Batch size limited to 1000 customers. For larger batches, use the async job endpoint."
        )

    predictions = []
    for i, customer in enumerate(customers):
        try:
            customer_dict = {
                "gender": customer.gender,
                "senior_citizen": customer.senior_citizen,
                "partner": customer.partner.value,
                "dependents": customer.dependents.value,
                "tenure_months": customer.tenure_months,
                "contract_type": customer.contract_type.value,
                "internet_service": customer.internet_service.value,
                "online_security": customer.online_security.value,
                "online_backup": customer.online_backup.value,
                "device_protection": customer.device_protection.value,
                "tech_support": customer.tech_support.value,
                "streaming_tv": customer.streaming_tv.value,
                "streaming_movies": customer.streaming_movies.value,
                "phone_service": customer.phone_service.value,
                "multiple_lines": customer.multiple_lines.value,
                "monthly_charges": customer.monthly_charges,
                "total_charges": customer.total_charges,
                "payment_method": customer.payment_method.value,
                "paperless_billing": customer.paperless_billing.value,
            }
            result = loader.predict(customer_dict)
            predictions.append(PredictionResponse(**result))
        except Exception as e:
            print(f"[BATCH] Error on customer {i}: {e}")
            continue

    # ── Aggregate metrics ──
    persuadables = [p for p in predictions if p.quadrant == "Persuadable"]
    sleeping_dogs = [p for p in predictions if p.quadrant == "Sleeping Dog"]

    total_revenue = sum(p.value_at_risk for p in predictions)
    persuadable_revenue = sum(p.value_at_risk for p in persuadables)

    print(f"[BATCH] Scored {len(predictions)} customers. "
          f"Persuadables: {len(persuadables)}, "
          f"Revenue at risk: ${total_revenue:,.0f}")

    return BatchPredictionResponse(
        total_customers=len(predictions),
        total_revenue_at_risk=round(total_revenue, 2),
        persuadable_count=len(persuadables),
        persuadable_revenue=round(persuadable_revenue, 2),
        sleeping_dog_count=len(sleeping_dogs),
        predictions=predictions,
    )


# ─────────────────────────────────────────────────────────────────────
# RUN (for development)
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\nStarting Retention Safeguard API...")
    print("Visit http://localhost:8000/docs for interactive documentation\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)