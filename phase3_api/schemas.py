"""
=============================================================================
PHASE 3 | Script 1 of 3: schemas.py
=============================================================================
Proactive Retention & Revenue Safeguard System
Author: Utsav Khadka (github.com/UtsavKhadka-Analyst)

PURPOSE:
    Define Pydantic v2 models for API request/response validation.
    These schemas act as CONTRACTS between the API and its consumers.

WHY PYDANTIC?
    1. Automatic input validation (wrong types → clear error messages)
    2. Auto-generated API documentation (Swagger/OpenAPI)
    3. Type safety — catches bugs before they hit your model
    4. Industry standard for FastAPI applications

BUSINESS CONTEXT:
    The API receives customer data from upstream systems (CRM, billing)
    and returns actionable insights for the marketing team:
    - Churn risk probability
    - Uplift score (can we save them?)
    - Quadrant classification (Persuadable, Sure Thing, etc.)
    - Dollar value at risk


=============================================================================
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


# ─────────────────────────────────────────────────────────────────────
# ENUMS — Constrain categorical inputs to valid values
# ─────────────────────────────────────────────────────────────────────

class ContractType(str, Enum):
    """Valid contract types from the Telco dataset."""
    MONTH_TO_MONTH = "Month-to-month"
    ONE_YEAR = "One year"
    TWO_YEAR = "Two year"


class InternetService(str, Enum):
    DSL = "DSL"
    FIBER_OPTIC = "Fiber optic"
    NO = "No"


class YesNo(str, Enum):
    YES = "Yes"
    NO = "No"


class PaymentMethod(str, Enum):
    ELECTRONIC_CHECK = "Electronic check"
    MAILED_CHECK = "Mailed check"
    BANK_TRANSFER = "Bank transfer (automatic)"
    CREDIT_CARD = "Credit card (automatic)"


class ServiceOption(str, Enum):
    """Services that depend on internet subscription."""
    YES = "Yes"
    NO = "No"
    NO_INTERNET = "No internet service"


class PhoneServiceOption(str, Enum):
    YES = "Yes"
    NO = "No"
    NO_PHONE = "No phone service"


class UpliftQuadrant(str, Enum):
    """The four uplift quadrants — core output of the causal model."""
    PERSUADABLE = "Persuadable"
    SURE_THING = "Sure Thing"
    LOST_CAUSE = "Lost Cause"
    SLEEPING_DOG = "Sleeping Dog"


# ─────────────────────────────────────────────────────────────────────
# REQUEST SCHEMA — What the API accepts
# ─────────────────────────────────────────────────────────────────────

class CustomerInput(BaseModel):
    """
    Input schema for a single customer prediction.

    Every field maps to a column in the Telco dataset.
    Pydantic validates types and constraints BEFORE the data
    reaches the model — this is your first line of defense
    against garbage-in-garbage-out.

    DESIGN DECISION:
        We require all fields (no Optional) because the uplift model
        needs every feature to make accurate predictions. Missing
        features would silently degrade prediction quality.
    """

    # ── Demographics ──
    gender: str = Field(
        ...,
        description="Customer gender",
        examples=["Male"]
    )
    senior_citizen: int = Field(
        ...,
        ge=0, le=1,
        description="1 if customer is 65+, 0 otherwise",
        examples=[0]
    )
    partner: YesNo = Field(
        ...,
        description="Whether customer has a partner",
        examples=["Yes"]
    )
    dependents: YesNo = Field(
        ...,
        description="Whether customer has dependents",
        examples=["No"]
    )

    # ── Account Info ──
    tenure_months: int = Field(
        ...,
        ge=0, le=72,
        description="Number of months the customer has been with the company",
        examples=[24]
    )
    contract_type: ContractType = Field(
        ...,
        description="Type of contract",
        examples=["Month-to-month"]
    )

    # ── Services ──
    internet_service: InternetService = Field(
        ...,
        description="Type of internet service",
        examples=["Fiber optic"]
    )
    online_security: ServiceOption = Field(
        ...,
        description="Whether customer has online security add-on",
        examples=["No"]
    )
    online_backup: ServiceOption = Field(
        ...,
        description="Whether customer has online backup add-on",
        examples=["Yes"]
    )
    device_protection: ServiceOption = Field(
        ...,
        description="Whether customer has device protection add-on",
        examples=["No"]
    )
    tech_support: ServiceOption = Field(
        ...,
        description="Whether customer has tech support add-on",
        examples=["No"]
    )
    streaming_tv: ServiceOption = Field(
        ...,
        description="Whether customer has streaming TV add-on",
        examples=["Yes"]
    )
    streaming_movies: ServiceOption = Field(
        ...,
        description="Whether customer has streaming movies add-on",
        examples=["Yes"]
    )
    phone_service: YesNo = Field(
        ...,
        description="Whether customer has phone service",
        examples=["Yes"]
    )
    multiple_lines: PhoneServiceOption = Field(
        default=PhoneServiceOption.NO,
        description="Whether customer has multiple phone lines",
        examples=["No"]
    )

    # ── Billing ──
    monthly_charges: float = Field(
        ...,
        ge=0, le=200,
        description="Monthly charge amount in dollars",
        examples=[70.35]
    )
    total_charges: float = Field(
        ...,
        ge=0,
        description="Total charges to date in dollars",
        examples=[1397.47]
    )
    payment_method: PaymentMethod = Field(
        ...,
        description="How the customer pays",
        examples=["Electronic check"]
    )
    paperless_billing: YesNo = Field(
        ...,
        description="Whether customer uses paperless billing",
        examples=["Yes"]
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "senior_citizen": 0,
                "partner": "Yes",
                "dependents": "No",
                "tenure_months": 12,
                "contract_type": "Month-to-month",
                "internet_service": "Fiber optic",
                "online_security": "No",
                "online_backup": "No",
                "device_protection": "Yes",
                "tech_support": "No",
                "streaming_tv": "Yes",
                "streaming_movies": "No",
                "phone_service": "Yes",
                "multiple_lines": "No",
                "monthly_charges": 79.85,
                "total_charges": 958.20,
                "payment_method": "Electronic check",
                "paperless_billing": "Yes",
            }
        }
    }


# ─────────────────────────────────────────────────────────────────────
# RESPONSE SCHEMAS — What the API returns
# ─────────────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """
    Response for a single customer prediction.

    BUSINESS CONTEXT:
        This response gives the marketing manager EVERYTHING they need
        to make a decision about this customer:

        1. churn_risk → "How likely is this customer to leave?"
        2. uplift_score → "Can we change their mind with an offer?"
        3. quadrant → "What category do they fall into?"
        4. value_at_risk → "How much money is at stake?"
        5. recommendation → "What should we actually DO?"

        The recommendation field is the KEY differentiator.
        Most ML APIs return numbers. This API returns ACTIONS.
    """
    customer_id: Optional[str] = Field(
        None,
        description="Customer identifier (if provided)"
    )
    churn_risk: float = Field(
        ...,
        ge=0, le=1,
        description="Probability of churn without intervention (μ₀)"
    )
    churn_risk_with_offer: float = Field(
        ...,
        ge=0, le=1,
        description="Probability of churn WITH intervention (μ₁)"
    )
    uplift_score: float = Field(
        ...,
        description="Treatment effect: μ₀ - μ₁. Positive = offer helps."
    )
    quadrant: UpliftQuadrant = Field(
        ...,
        description="Uplift quadrant classification"
    )
    customer_lifetime_value: float = Field(
        ...,
        ge=0,
        description="Estimated CLV in dollars"
    )
    value_at_risk: float = Field(
        ...,
        description="CLV × uplift_score — dollar impact of targeting"
    )
    recommendation: str = Field(
        ...,
        description="Plain-English action recommendation"
    )
    confidence: str = Field(
        ...,
        description="Confidence level of the prediction (high/medium/low)"
    )


class HealthResponse(BaseModel):
    """Health check response with model metadata."""
    status: str = Field(..., description="API status")
    model_type: str = Field(..., description="Type of uplift model loaded")
    n_training_samples: int = Field(..., description="Training set size")
    training_date: str = Field(..., description="When the model was trained")
    n_features: int = Field(..., description="Number of input features")


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction endpoint."""
    total_customers: int
    total_revenue_at_risk: float
    persuadable_count: int
    persuadable_revenue: float
    sleeping_dog_count: int
    predictions: list[PredictionResponse]