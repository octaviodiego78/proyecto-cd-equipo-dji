# Gold Price Prediction - Docker Deployment

This directory contains the Docker configuration to run both the backend API and frontend UI in containers.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Localhost                      │
│                                                 │
│  ┌──────────────┐         ┌─────────────────┐  │
│  │   Frontend   │────────▶│    Backend      │  │
│  │  (Streamlit) │         │   (FastAPI)     │  │
│  │  Port: 8501  │         │   Port: 8000    │  │
│  └──────────────┘         └─────────────────┘  │
│                                    │            │
│                                    ▼            │
│                           ┌─────────────────┐  │
│                           │  Data/Artifacts │  │
│                           │  (Volume Mount) │  │
│                           └─────────────────┘  │
└─────────────────────────────────────────────────┘
```

## Prerequisites

1. **Docker Desktop** installed and running
2. **Trained model artifacts** in `../data/processed/`:
   - `scaler.pkl`
   - `feature_columns.json`
   - `model_metadata.json`
3. **Environment variables** (optional):
   - Create `../.env` file with Databricks credentials if using MLflow model registry

## Quick Start

### Option 1: Use the start script (Recommended)

```bash
cd src/
./start.sh
```

### Option 2: Manual Docker Compose

```bash
cd src/
docker compose up --build
```

### Option 3: Run services individually

```bash
# Start backend only
docker compose up backend

# Start frontend only
docker compose up frontend
```

## Accessing the Services

Once the containers are running:

- **Frontend UI**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Docker Compose Configuration

The `docker-compose.yaml` file defines:

### Backend Service
- **Container**: `gold-prediction-backend`
- **Port**: 8000
- **Health Check**: Curl check on `/health` endpoint
- **Volumes**: 
  - `../data/processed` (model artifacts, read-only)
  - `../.env` (Databricks credentials, read-only)

### Frontend Service
- **Container**: `gold-prediction-frontend`
- **Port**: 8501
- **Depends On**: Backend service (waits for healthy status)
- **Environment**: `API_URL=http://backend:8000`

### Network
- Both services communicate via a bridge network: `gold-prediction-network`

## Useful Commands

### View logs
```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f backend
docker compose logs -f frontend
```

### Stop services
```bash
docker compose down
```

### Restart services
```bash
docker compose restart

# Restart specific service
docker compose restart backend
```

### Rebuild containers
```bash
docker compose up --build
```

### Check service status
```bash
docker compose ps
```

### Execute commands in running container
```bash
# Backend
docker compose exec backend bash

# Frontend
docker compose exec frontend bash
```

## Troubleshooting

### Backend fails to start

1. **Check if model artifacts exist**:
   ```bash
   ls -la ../data/processed/
   ```
   Should show: `scaler.pkl`, `feature_columns.json`, `model_metadata.json`

2. **Check backend logs**:
   ```bash
   docker compose logs backend
   ```

3. **Verify .env file** (if using Databricks):
   ```bash
   cat ../.env
   ```

### Frontend cannot connect to backend

1. **Check if backend is healthy**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Verify network configuration**:
   ```bash
   docker network ls
   docker network inspect gold-prediction-network
   ```

3. **Check frontend logs**:
   ```bash
   docker compose logs frontend
   ```

### Port conflicts

If ports 8000 or 8501 are already in use, modify `docker-compose.yaml`:

```yaml
services:
  backend:
    ports:
      - "8001:8000"  # Change host port
  
  frontend:
    ports:
      - "8502:8501"  # Change host port
```

### Container crashes immediately

1. **Check Docker resources** (Memory/CPU in Docker Desktop settings)
2. **Review build logs**:
   ```bash
   docker compose build --no-cache
   ```

## Development

### Rebuild after code changes

```bash
# Rebuild and restart
docker compose up --build

# Force recreate containers
docker compose up --force-recreate
```

### Test API directly

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"predict_tomorrow": true}'
```

## Production Considerations

Before deploying to production:

1. **Update CORS settings** in `backend/api.py`:
   ```python
   allow_origins=["https://your-frontend-domain.com"]
   ```

2. **Add authentication** to API endpoints

3. **Use environment-specific configuration**:
   ```bash
   docker compose -f docker-compose.prod.yaml up
   ```

4. **Set resource limits** in `docker-compose.yaml`:
   ```yaml
   services:
     backend:
       deploy:
         resources:
           limits:
             cpus: '1'
             memory: 2G
   ```

5. **Use secrets management** instead of `.env` file

6. **Enable HTTPS** with reverse proxy (nginx/traefik)

## File Structure

```
src/
├── docker-compose.yaml      # Orchestration configuration
├── start.sh                 # Quick start script
├── README.md               # This file
├── backend/
│   ├── Dockerfile          # Backend container definition
│   ├── requirements.txt    # Python dependencies
│   ├── api.py             # FastAPI application
│   ├── model_utils.py     # Model loading utilities
│   ├── preprocessing.py   # Data preprocessing
│   └── data_fetcher.py    # Yahoo Finance data fetcher
└── frontend/
    ├── Dockerfile          # Frontend container definition
    ├── requirements.txt    # Python dependencies
    └── app.py             # Streamlit application
```

## License

This project is part of the ITESO Computer Science program.

