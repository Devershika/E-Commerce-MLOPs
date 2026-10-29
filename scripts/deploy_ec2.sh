#!/usr/bin/env bash
# ==============================================================================
# AWS EC2 Deployment & Service Keep-Alive Script
# Project: E-Commerce Demand Forecasting MLOps
# ==============================================================================
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-ghcr.io/devershika/e-commerce-mlops/demand-forecast-api:latest}"
CONTAINER_NAME="demand_forecast_service"
PORT="${PORT:-8000}"

echo "=========================================================="
echo " Starting Deployment for: ${IMAGE_NAME}"
echo "=========================================================="

# 1. Ensure Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo "Docker not found! Installing Docker..."
    sudo apt-get update -y
    sudo apt-get install -y ca-certificates curl gnupg lsb-release
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    sudo systemctl enable docker
    sudo systemctl start docker
fi

# 2. Login to Registry if credentials provided in environment
if [[ -n "${REGISTRY_USER:-}" && -n "${REGISTRY_TOKEN:-}" ]]; then
    echo "Logging into container registry..."
    echo "${REGISTRY_TOKEN}" | docker login ghcr.io -u "${REGISTRY_USER}" --password-stdin
fi

# 3. Pull latest image from registry
echo "Pulling latest image: ${IMAGE_NAME}..."
docker pull "${IMAGE_NAME}"

# 4. Stop and remove existing container if running
if docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}\$"; then
    echo "Stopping existing container: ${CONTAINER_NAME}..."
    docker stop "${CONTAINER_NAME}" || true
    echo "Removing existing container: ${CONTAINER_NAME}..."
    docker rm "${CONTAINER_NAME}" || true
fi

# 5. Launch container with restart: always policy
echo "Launching container with restart: always on port ${PORT}..."
docker run -d \
    --name "${CONTAINER_NAME}" \
    --restart always \
    -p "${PORT}:8000" \
    "${IMAGE_NAME}"

# 6. Verify healthcheck
echo "Waiting for service to be healthy..."
for i in {1..12}; do
    if curl -s -f "http://localhost:${PORT}/health" > /dev/null; then
        echo " Service is healthy and serving predictions on port ${PORT}!"
        exit 0
    fi
    echo "Waiting for healthcheck (attempt ${i}/12)..."
    sleep 3
done

echo "⚠️ Container started, but healthcheck did not respond within 36s. Check logs:"
docker logs --tail 50 "${CONTAINER_NAME}"
exit 1
