---
title: Gold Price Prediction API
emoji: 🔧
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Gold Price Prediction API (Backend)

FastAPI backend for gold price prediction using MLflow models from Databricks.

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Make a prediction

## Configuration

Add these secrets in Space Settings:
- `DATABRICKS_HOST` - Your Databricks workspace URL
- `DATABRICKS_TOKEN` - Your Databricks access token

