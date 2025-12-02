# Gold Price Prediction Service

A production-ready machine learning service for predicting tomorrow's gold prices using historical gold and S&P 500 data. The system leverages TensorFlow neural networks (MLP/CNN/LSTM), MLflow for model management, Prefect for orchestration, and Docker for containerization.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Training Pipeline (Prefect)                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │  Load    │──▶│ Feature  │──▶│  Train   │──▶│ Register │    │
│  │  Data    │   │Engineer  │   │  Models  │   │  Champion│    │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘    │
│         │             │                               │          │
│         ▼             ▼                               ▼          │
│   data/raw/    data/processed/              MLflow Registry     │
│   - gold_data.csv  - scaler.pkl           (Databricks)          │
│   - sp500.csv      - feature_columns.json                       │
│                    - model_metadata.json                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Prediction Service (Docker)                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              FastAPI Backend (Port 8000)                   │ │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐ │ │
│  │  │  Fetch   │──▶│ Feature  │──▶│  Scale   │──▶│Predict │ │ │
│  │  │Yahoo     │   │Engineer  │   │ Features │   │ (Model)│ │ │
│  │  │Finance   │   │          │   │          │   │        │ │ │
│  │  └──────────┘   └──────────┘   └──────────┘   └────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │            Streamlit Frontend (Port 8501)                  │ │
│  │           🏆 Gold Price Prediction UI                      │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 📋 Features

- **Live Data Integration**: Fetches real-time gold (GC=F) and S&P 500 (^GSPC) prices from Yahoo Finance
- **Advanced ML Models**: Supports MLP, CNN, and LSTM architectures with hyperparameter optimization
- **MLflow Integration**: Model versioning, tracking, and registry with Databricks Unity Catalog
- **Prefect Orchestration**: Automated training pipeline with tasks and flows
- **Production-Ready API**: FastAPI backend with health checks and proper error handling
- **User-Friendly Interface**: Streamlit frontend for easy predictions
- **Containerized Deployment**: Docker and docker-compose for consistent environments

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Databricks account with MLflow access
- Environment variables configured (see `.env.example`)

### 1. Environment Setup

Create a `.env` file in the project root:

```bash
DATABRICKS_HOST=https://your-workspace.databricks.com
DATABRICKS_TOKEN=your-databricks-token
```

### 2. Training Pipeline

#### Run Prefect Training Flow

```bash
# Install dependencies
pip install -r src/pipelines/requirements.txt

# Run the training pipeline
python src/pipelines/train_pipeline.py
```

**Pipeline Tasks:**
1. Load and prepare data from CSV files
2. Feature engineering (lags, moving averages, volatility)
3. Train baseline models (MLP, CNN, LSTM)
4. Hyperparameter optimization with Hyperopt
5. Select best model based on MAPE
6. Register champion model to MLflow
7. Save artifacts (scaler, feature columns, metadata)

**Outputs:**
- `data/processed/scaler.pkl` - Fitted StandardScaler
- `data/processed/feature_columns.json` - Feature column names
- `data/processed/model_metadata.json` - Model type and name
- Model registered in Databricks MLflow Registry with "champion" alias

#### Prefect Flow Logs

The pipeline provides detailed logging for each task:
- Data loading statistics
- Feature engineering progress
- Training metrics (MAPE, RMSE, MAE, R²)
- Model registration details

### 3. Running the Service

#### Option A: Docker Compose (Recommended)

```bash
# Build and start services
docker-compose up --build

# Access the services:
# - Frontend: http://localhost:8501
# - Backend API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

#### Option B: Local Development

**Terminal 1 - Backend:**
```bash
cd src/backend
pip install -r requirements.txt
python api.py
```

**Terminal 2 - Frontend:**
```bash
cd src/frontend
pip install -r requirements.txt
streamlit run app.py
```

### 4. Making Predictions

#### Via Streamlit UI
1. Open http://localhost:8501
2. Click "🔮 Predict Tomorrow's Gold Price"
3. View prediction, date, and model details

#### Via API

**Health Check:**
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true,
  "feature_cols_loaded": true
}
```

**Make Prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"predict_tomorrow": true}'
```

Response:
```json
{
  "prediction": 1950.75,
  "predicted_date": "2024-12-03",
  "today_date": "2024-12-02",
  "model_name": "workspace.default.equipo_dji_gold_prediction_model",
  "model_type": "MLP"
}
```

## 🐳 Docker Configuration

### Backend Dockerfile (`src/backend/Dockerfile`)

- Base image: `python:3.11-slim`
- Installs: TensorFlow, FastAPI, MLflow, yfinance, pandas, scikit-learn
- Copies: Backend code, processed data (scaler, features), `.env`
- Exposes: Port 8000
- Health check: `/health` endpoint

### Frontend Dockerfile (`src/frontend/DockerFile`)

- Base image: `python:3.11-slim`
- Installs: Streamlit, requests
- Copies: Frontend application
- Exposes: Port 8501
- Environment: `API_URL=http://backend:8000`

### Docker Compose Services

**Backend:**
- Ports: 8000:8000
- Volumes: `data/processed` (read-only), `src/backend` (read-only)
- Health check: 30s interval, 40s start period
- Environment: Databricks credentials, artifact paths

**Frontend:**
- Ports: 8501:8501
- Depends on: Backend (waits for healthy status)
- Environment: `API_URL=http://backend:8000`

## 📊 API Endpoints

### GET /health

Health check endpoint to verify service status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true,
  "feature_cols_loaded": true
}
```

### POST /predict

Predict tomorrow's gold price using live data.

**Request Body:**
```json
{
  "predict_tomorrow": true
}
```

**Response:**
```json
{
  "prediction": 1950.75,
  "predicted_date": "2024-12-03",
  "today_date": "2024-12-02",
  "model_name": "workspace.default.equipo_dji_gold_prediction_model",
  "model_type": "MLP"
}
```

**Prediction Flow:**
1. Fetch last 30 days of gold and S&P 500 prices from Yahoo Finance
2. Merge and sort data by date
3. Apply feature engineering (same as training):
   - Lag features (1-2 days) for gold and S&P 500
   - 5-day moving averages
   - 5-day volatility (gold)
   - S&P 500 returns (lag 1)
4. Extract features for most recent date
5. Scale features using saved `StandardScaler`
6. Reshape based on model type (MLP, CNN, LSTM)
7. Predict tomorrow's price
8. Return prediction with metadata

## 🚢 Deploying to HuggingFace Spaces

### Step 1: Prepare Repository

Create a HuggingFace Space with Docker SDK:

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Name: `gold-price-prediction`
4. SDK: **Docker**
5. Hardware: CPU Basic (or GPU if available)

### Step 2: Repository Structure

Your Space should have this structure:
```
.
├── Dockerfile              # Main Dockerfile for HF Spaces
├── docker-compose.yaml     # (Optional, for local testing)
├── src/
│   ├── backend/
│   │   ├── api.py
│   │   ├── preprocessing.py
│   │   ├── model_utils.py
│   │   └── data_fetcher.py
│   └── frontend/
│       └── app.py
├── data/
│   └── processed/
│       ├── scaler.pkl
│       ├── feature_columns.json
│       └── model_metadata.json
└── requirements.txt
```

### Step 3: HuggingFace Dockerfile

Create a `Dockerfile` for HuggingFace Spaces (combines backend and frontend):

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY src/backend/requirements.txt backend-requirements.txt
COPY src/frontend/requirements.txt frontend-requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r backend-requirements.txt && \
    pip install --no-cache-dir -r frontend-requirements.txt

# Copy application code
COPY src/ /app/src/
COPY data/processed/ /app/data/processed/

# Copy .env file (add secrets in HF Spaces settings instead)
# COPY .env /app/.env

# Set environment variables
ENV PYTHONPATH=/app
ENV API_URL=http://localhost:8000

# Expose ports
EXPOSE 8000 8501

# Create startup script to run both services
RUN echo '#!/bin/bash\n\
uvicorn src.backend.api:app --host 0.0.0.0 --port 8000 &\n\
sleep 10\n\
streamlit run src/frontend/app.py --server.port=8501 --server.address=0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
```

### Step 4: Add Secrets in HuggingFace Spaces

1. Go to your Space settings
2. Add Repository Secrets:
   - `DATABRICKS_HOST`: Your Databricks workspace URL
   - `DATABRICKS_TOKEN`: Your Databricks access token

### Step 5: Push to HuggingFace

```bash
# Clone your Space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/gold-price-prediction
cd gold-price-prediction

# Copy files
cp -r src/ .
cp -r data/processed/ .
cp Dockerfile .

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

### Step 6: Monitor Deployment

- HuggingFace will automatically build and deploy
- Check build logs in the Space's "Logs" tab
- Once deployed, access at: `https://huggingface.co/spaces/YOUR_USERNAME/gold-price-prediction`

### Screenshots Required

Capture screenshots of:
1. **Streamlit UI**: Showing prediction result
2. **API Documentation**: `http://localhost:8000/docs`
3. **HuggingFace Space**: Running service
4. **Prefect Flow Logs**: Training pipeline execution

### Troubleshooting HuggingFace Deployment

**Issue: Model loading fails**
- Ensure Databricks secrets are correctly set
- Check that `data/processed/` artifacts are included
- Verify network access to Databricks

**Issue: Services don't start**
- Check logs for port conflicts
- Ensure startup script waits for backend to be ready
- Verify Python dependencies are compatible

**Issue: Out of memory**
- Upgrade to larger hardware tier
- Reduce model size or use quantization
- Consider serving only backend API

## 🛠️ Project Structure

```
proyecto-cd-equipo-dji/
├── src/
│   ├── backend/
│   │   ├── api.py                  # FastAPI application
│   │   ├── preprocessing.py        # Feature engineering utilities
│   │   ├── model_utils.py          # MLflow model loading
│   │   ├── data_fetcher.py         # Yahoo Finance integration
│   │   ├── requirements.txt        # Backend dependencies
│   │   └── Dockerfile              # Backend container
│   ├── frontend/
│   │   ├── app.py                  # Streamlit application
│   │   ├── requirements.txt        # Frontend dependencies
│   │   └── DockerFile              # Frontend container
│   └── pipelines/
│       └── train_pipeline.py       # Prefect training flow
├── data/
│   ├── raw/
│   │   ├── gold_data.csv           # Historical gold prices
│   │   └── sp500.csv               # Historical S&P 500 data
│   └── processed/
│       ├── scaler.pkl              # Fitted StandardScaler
│       ├── feature_columns.json    # Feature names
│       └── model_metadata.json     # Model type and name
├── docker-compose.yaml             # Multi-container orchestration
├── .env                            # Environment variables (not in git)
└── README.md                       # This file
```

## 📦 Dependencies

### Backend
- `fastapi>=0.109.0` - Modern web framework
- `uvicorn>=0.27.0` - ASGI server
- `tensorflow>=2.15.0` - Neural network models
- `mlflow>=2.10.0` - Model tracking and registry
- `yfinance>=0.2.35` - Yahoo Finance data
- `pandas>=2.1.0` - Data manipulation
- `scikit-learn>=1.3.0` - Preprocessing and metrics

### Frontend
- `streamlit>=1.30.0` - Interactive web app
- `requests>=2.31.0` - HTTP client

### Training Pipeline
- `prefect>=3.0.0` - Workflow orchestration
- `hyperopt>=0.2.7` - Hyperparameter optimization
- All backend dependencies

## 🔧 Configuration

### Environment Variables

**Required:**
- `DATABRICKS_HOST` - Databricks workspace URL
- `DATABRICKS_TOKEN` - Databricks personal access token

**Optional:**
- `SCALER_PATH` - Path to scaler file (default: `data/processed/scaler.pkl`)
- `FEATURE_COLS_PATH` - Path to feature columns (default: `data/processed/feature_columns.json`)
- `MODEL_METADATA_PATH` - Path to model metadata (default: `data/processed/model_metadata.json`)
- `API_URL` - Backend API URL for frontend (default: `http://localhost:8000`)

## 🧪 Testing

### Test Training Pipeline
```bash
python src/pipelines/train_pipeline.py
```

### Test Backend API
```bash
# Start backend
python src/backend/api.py

# In another terminal, test endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"predict_tomorrow": true}'
```

### Test Frontend
```bash
streamlit run src/frontend/app.py
```

### Test Docker Services
```bash
docker-compose up --build
```

## 📝 Logs and Monitoring

### View Container Logs
```bash
# All services
docker-compose logs -f

# Backend only
docker-compose logs -f backend

# Frontend only
docker-compose logs -f frontend
```

### Monitor Prefect Flows
```bash
# View flow runs
prefect deployment run 'Gold Price Prediction Training Pipeline'

# Check flow status
prefect flow-run ls
```

### MLflow Tracking
- View experiments in Databricks MLflow UI
- Track metrics: MAPE, RMSE, MAE, R²
- Compare model versions
- View artifacts and parameters

## 🛑 Stopping Services

```bash
# Stop Docker containers
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Stop local processes
# Press Ctrl+C in terminal running the service
```

## 📄 License

This project is part of the ITESO Ciencia de Datos program.

## 👥 Team

Equipo DJI - ITESO Proyecto Ciencia de Datos

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues or questions:
- Check logs: `docker-compose logs`
- Verify environment variables
- Ensure Databricks credentials are correct
- Check network connectivity to Yahoo Finance

---

**Note**: This service uses live data from Yahoo Finance. Predictions are for educational purposes only and should not be used for financial decisions.

