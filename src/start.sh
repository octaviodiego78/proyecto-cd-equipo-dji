#!/bin/bash

# Script to start both backend and frontend services using Docker Compose
# This script should be run from the src/ directory

set -e

echo "Starting Gold Price Prediction Services..."echo ""

echo "Building and starting containers..."
echo ""

# Build and start services
docker compose up --build -d


# Wait a bit for services to initialize
sleep 5

# Check backend health
echo "Checking backend..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "Backend is healthy"
else
    echo "Backend is starting up... (this may take a minute)"
fi

