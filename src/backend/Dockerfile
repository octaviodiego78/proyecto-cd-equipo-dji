FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (needed for TensorFlow and building packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . /app/src/backend/

# Copy data/processed directory for scaler and feature columns
COPY ../../data/processed /app/data/processed

# Copy .env file if it exists (for Databricks credentials)
COPY ../../.env /app/.env

# Set Python path to find modules
ENV PYTHONPATH=/app

# Expose port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI application
CMD ["uvicorn", "src.backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
