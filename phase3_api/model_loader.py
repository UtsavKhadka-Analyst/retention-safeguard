"""
=============================================================================
PHASE 3 | Script 2 of 3: model_loader.py
=============================================================================
Proactive Retention & Revenue Safeguard System
Author: Utsav Khadka (github.com/UtsavKhadka-Analyst)

PURPOSE:
    Handle model loading, feature engineering, and inference logic.
    This module is the BRIDGE between raw API input and model predictions.

DESIGN PATTERN — SINGLETON MODEL LOADER:
    We load models ONCE when the API starts, not on every request.
    This avoids the overhead of deserializing pickle files on every call.
    In production, this pattern is critical for low-latency serving.

    The ModelLoader class:
    1. Loads the T-Learner bundle (control + treatment models)
    2. Transforms raw customer data into model-ready features
    3. Runs inference and computes uplift scores
    4. Classifies into quadrants and generates recommendations

INTERVIEW PREP:
    Q: "How do you serve ML models in production?"
    A: "I separate model loading from inference. Models are loaded once
       at startup and cached in memory. Each request only runs the
       feature transformation and predict calls, keeping latency low.
       For higher scale, I'd use model servers like TorchServe or
       Triton, but for this throughput, in-process serving is sufficient."
=============================================================================
"""

import pickle
import numpy as np
import os
from typing import Optional


class ModelLoader:
    """
    Singleton model loader for the T-Learner uplift models.

    ARCHITECTURE:
        ┌──────────┐    ┌──────────────┐    ┌─────────────┐
        │ Raw JSON │ →  │ Feature Eng  │ →  │ Model 0 + 1 │ → Uplift Score
        │ (API)    │    │ (encode,     │    │ (predict    │ → Quadrant
        └──────────┘    │  compute)    │    │  proba)     │ → Recommendation
                        └──────────────┘    └─────────────┘

    WHY A CLASS?
        Encapsulates all model-related state (models, encoders, feature names)
        in one object. The API (main.py) just calls loader.predict(data)
        without knowing any model internals. This is the Single Responsibility
        Principle in action.
    """

    def __init__(self, model_dir: str):
        """
        Load model artifacts from disk.

        Args:
            model_dir: Path to the artifacts/ directory from Phase 2
        """
        self.model_dir = model_dir
        self.model_control = None
        self.model_treatment = None
        self.label_encoders = None
        self.feature_names = None
        self.model_metadata = {}
        self._loaded = False

    def load(self) -> None:
        """
        Load all model artifacts into memory.

        Called ONCE at API startup. After this, all predict() calls
        use the in-memory models — no disk I/O per request.

        WHAT WE LOAD:
            - model_control: GBM trained on control group → predicts P(churn | no offer)
            - model_treatment: GBM trained on treatment group → predicts P(churn | offer)
            - label_encoders: Maps categorical strings → integers
            - feature_names: Ordered list of feature columns (order matters!)
        """
        print("[ModelLoader] Loading model artifacts...")

        bundle_path = os.path.join(self.model_dir, "t_learner_bundle.pkl")

        if not os.path.exists(bundle_path):
            raise FileNotFoundError(
                f"Model bundle not found at {bundle_path}. "
                f"Run Phase 2 (t_learner.py) first."
            )

        with open(bundle_path, "rb") as f:
            bundle = pickle.load(f)

        self.model_control = bundle["model_control"]
        self.model_treatment = bundle["model_treatment"]
        self.label_encoders = bundle["label_encoders"]
        self.feature_names = bundle["feature_names"]
        self.model_metadata = {
            "model_type": bundle.get("model_type", "T-Learner"),
            "training_date": bundle.get("training_date", "unknown"),
            "n_training_samples": bundle.get("n_training_samples", 0),
            "n_features": len(self.feature_names),
        }
        self._loaded = True

        print(f"[ModelLoader] Loaded {self.model_metadata['model_type']} "
              f"({self.model_metadata['n_features']} features, "
              f"{self.model_metadata['n_training_samples']} training samples)")

    def is_loaded(self) -> bool:
        return self._loaded

    def transform_input(self, customer_data: dict) -> np.ndarray:
        """
        Transform raw customer JSON into a feature vector for the model.

        THIS IS WHERE MOST PRODUCTION BUGS LIVE.
        The feature vector must have:
        1. The SAME features as training (same columns, same order)
        2. The SAME encoding (same label mapping)
        3. Computed features (num_premium_services, CLV, propensity_score)

        If any of these are wrong, the model produces silent garbage.

        INTERVIEW PREP:
            Q: "What's the most common cause of model degradation in production?"
            A: "Feature/training skew. The model was trained on features processed
               one way, but the serving pipeline processes them differently.
               I enforce consistency by using the SAME label encoders saved
               during training and computing derived features identically."
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load() first.")

        # ── Compute derived features ──
        # These must match EXACTLY how they were computed in Phase 1/2

        # num_premium_services: count of add-on services
        premium_services = [
            customer_data.get("online_security", "No"),
            customer_data.get("online_backup", "No"),
            customer_data.get("device_protection", "No"),
            customer_data.get("tech_support", "No"),
            customer_data.get("streaming_tv", "No"),
            customer_data.get("streaming_movies", "No"),
        ]
        num_premium = sum(1 for s in premium_services if s == "Yes")

        # CLV (same formula as Phase 1)
        tenure = customer_data["tenure_months"]
        monthly = customer_data["monthly_charges"]
        contract = customer_data["contract_type"]

        contract_map = {"Month-to-month": 6, "One year": 14, "Two year": 26}
        tenure_factor = np.log1p(tenure) * 3
        contract_mult = contract_map.get(contract, 6)
        clv = monthly * (tenure_factor + contract_mult)

        # Propensity score placeholder (estimated from features)
        # In production, this would come from the propensity model.
        # Here we use a simple heuristic matching Phase 1's logic.
        risk_signal = (
            (1 if tenure < 24 else 0) * 0.3
            + (1 if monthly > 70 else 0) * 0.2
            + (1 if contract == "Month-to-month" else 0) * 0.3
        )
        propensity_score = min(max(risk_signal, 0.05), 0.95)

        # ── Build feature dictionary matching training order ──
        feature_dict = {
            "gender": customer_data["gender"],
            "senior_citizen": customer_data["senior_citizen"],
            "partner": customer_data["partner"],
            "dependents": customer_data["dependents"],
            "tenure_months": tenure,
            "contract_type": contract,
            "internet_service": customer_data["internet_service"],
            "online_security": customer_data["online_security"],
            "online_backup": customer_data["online_backup"],
            "device_protection": customer_data["device_protection"],
            "tech_support": customer_data["tech_support"],
            "streaming_tv": customer_data["streaming_tv"],
            "streaming_movies": customer_data["streaming_movies"],
            "phone_service": customer_data["phone_service"],
            "num_premium_services": num_premium,
            "monthly_charges": monthly,
            "total_charges": customer_data["total_charges"],
            "payment_method": customer_data["payment_method"],
            "paperless_billing": customer_data["paperless_billing"],
            "clv": clv,
            "propensity_score": propensity_score,
        }

        # ── Encode categoricals using the SAME encoders from training ──
        feature_vector = []
        for fname in self.feature_names:
            val = feature_dict.get(fname)

            if fname in self.label_encoders:
                le = self.label_encoders[fname]
                # Handle unseen categories gracefully
                if val in le.classes_:
                    val = le.transform([val])[0]
                else:
                    # Unseen category → use most frequent class (safe default)
                    val = 0
                    print(f"[WARNING] Unseen category '{val}' for feature '{fname}'. Using default.")

            feature_vector.append(float(val))

        return np.array([feature_vector]), clv, propensity_score

    def predict(self, customer_data: dict) -> dict:
        """
        Run the full prediction pipeline for a single customer.

        PIPELINE:
            1. Transform raw input → feature vector
            2. Model 0 predicts: P(churn | no offer) = μ₀
            3. Model 1 predicts: P(churn | offer) = μ₁
            4. Uplift = μ₀ - μ₁ (positive = offer helps)
            5. Classify into quadrant
            6. Compute value at risk
            7. Generate recommendation

        Returns a dictionary matching the PredictionResponse schema.
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load() first.")

        # ── Step 1: Transform ──
        X, clv, propensity = self.transform_input(customer_data)

        # ── Step 2-3: Predict churn probabilities ──
        mu_0 = self.model_control.predict_proba(X)[0, 1]   # P(churn | no offer)
        mu_1 = self.model_treatment.predict_proba(X)[0, 1]  # P(churn | offer)

        # ── Step 4: Uplift ──
        uplift = float(mu_0 - mu_1)

        # ── Step 5: Quadrant classification ──
        # Using same thresholds as Phase 2
        uplift_threshold = 0.01
        risk_threshold = 0.27  # Approximate median from training data

        if uplift > uplift_threshold and mu_0 > risk_threshold:
            quadrant = "Persuadable"
        elif uplift > uplift_threshold and mu_0 <= risk_threshold:
            quadrant = "Sure Thing"
        elif uplift < -uplift_threshold:
            quadrant = "Sleeping Dog"
        else:
            quadrant = "Lost Cause"

        # ── Step 6: Value at risk ──
        value_at_risk = round(clv * uplift, 2)

        # ── Step 7: Recommendation ──
        recommendation = self._generate_recommendation(
            quadrant, uplift, mu_0, clv, value_at_risk
        )

        # ── Step 8: Confidence level ──
        confidence = self._assess_confidence(uplift, mu_0)

        return {
            "churn_risk": round(float(mu_0), 4),
            "churn_risk_with_offer": round(float(mu_1), 4),
            "uplift_score": round(uplift, 4),
            "quadrant": quadrant,
            "customer_lifetime_value": round(float(clv), 2),
            "value_at_risk": value_at_risk,
            "recommendation": recommendation,
            "confidence": confidence,
        }

    def _generate_recommendation(self, quadrant: str, uplift: float,
                                  churn_risk: float, clv: float,
                                  value_at_risk: float) -> str:
        """
        Generate a plain-English recommendation for the marketing team.

        THIS IS WHAT MAKES THE API USEFUL TO NON-TECHNICAL USERS.
        A marketing manager doesn't care about uplift scores.
        They care about: "What should I DO with this customer?"

        Each quadrant maps to a specific action:
        - Persuadable → Target with retention offer
        - Sure Thing → Monitor but don't spend budget
        - Lost Cause → Deprioritize, focus resources elsewhere
        - Sleeping Dog → Do NOT contact under any circumstances
        """
        if quadrant == "Persuadable":
            return (
                f"HIGH PRIORITY: Target with retention offer. "
                f"This customer has a {churn_risk:.0%} chance of churning without intervention, "
                f"but the offer could reduce it by {uplift:.0%}. "
                f"Potential savings: ${value_at_risk:,.0f}."
            )
        elif quadrant == "Sure Thing":
            return (
                f"LOW PRIORITY: Customer is likely to stay regardless. "
                f"Churn risk is only {churn_risk:.0%}. "
                f"Save your budget for Persuadable customers."
            )
        elif quadrant == "Sleeping Dog":
            return (
                f"DO NOT CONTACT: Marketing outreach would INCREASE churn risk "
                f"by {abs(uplift):.0%}. This customer is better left alone. "
                f"Contacting them could destroy ${abs(value_at_risk):,.0f} in value."
            )
        else:  # Lost Cause
            return (
                f"DEPRIORITIZE: Customer has {churn_risk:.0%} churn risk and the offer "
                f"has minimal effect (uplift: {uplift:+.2%}). "
                f"Reallocate resources to Persuadable segment."
            )

    def _assess_confidence(self, uplift: float, churn_risk: float) -> str:
        """
        Assess prediction confidence based on score extremity.

        LOGIC:
            Strong uplift signals (far from zero) are more reliable
            than weak signals (near zero) where noise dominates.
        """
        if abs(uplift) > 0.15:
            return "high"
        elif abs(uplift) > 0.05:
            return "medium"
        else:
            return "low"