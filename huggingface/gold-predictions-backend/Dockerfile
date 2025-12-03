FROM python:3.11-slim

WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

# Copy requirements
COPY requirements.txt .

# Install Python dependencies with uv
RUN uv pip install --system --no-cache -r requirements.txt

# Copy backend code
COPY *.py ./

# Copy processed data (scaler, features, metadata)
COPY data/processed/ /app/data/processed/

# Set environment variables
ENV PYTHONPATH=/app

# Expose FastAPI port
EXPOSE 7860

# Start FastAPI on port 7860 (HuggingFace Spaces default)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]

