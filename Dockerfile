# =============================================================================
# Dockerfile — Retention Safeguard System
# =============================================================================
# Builds a container with both the FastAPI backend and Streamlit dashboard.
# Use docker-compose.yml for the recommended multi-service setup.
#
# BUILD:  docker build -t retention-safeguard .
# RUN:    docker run -p 8000:8000 -p 8501:8501 retention-safeguard
# =============================================================================

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose ports: 8000 (FastAPI) + 8501 (Streamlit)
EXPOSE 8000 8501

# Default: start the API
CMD ["uvicorn", "phase3_api.main:app", "--host", "0.0.0.0", "--port", "8000"]