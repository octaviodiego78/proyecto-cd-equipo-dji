---
title: Gold Price Prediction
emoji: 📈
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
---

# Gold Price Prediction (Frontend)

Streamlit frontend for gold price prediction. Connects to the backend API to make predictions.

## Backend API

Connected to: https://octaviodiego78-gold-predictions-backend.hf.space

The backend handles:
- Loading the ML model from Databricks MLflow
- Fetching live data from Yahoo Finance
- Making predictions using TensorFlow models (MLP/CNN/LSTM)

