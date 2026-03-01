# =============================================================================
# Dockerfile — Retention Safeguard System
# =============================================================================
# Builds a container with both the FastAPI backend and Streamlit dashboard.
# Use docker-compose.yml for the recommended multi-service setup.
#
# BUILD:  docker build -t retention-safeguard .
# RUN:    docker run -p 8000:8000 -p 8501:8501 retention-safeguard
# =============================================================================

FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python phase1_data/fetch_and_enrich_data.py && \
    python phase1_data/load_to_db.py && \
    python phase1_data/sql_queries.py && \
    python phase2_modeling/t_learner.py
EXPOSE 8000 8501
CMD ["uvicorn", "phase3_api.main:app", "--host", "0.0.0.0", "--port", "8000"]