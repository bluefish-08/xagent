#!/usr/bin/env bash
# Build and push Docker images to Docker Hub
set -euo pipefail

REGISTRY="xprobe"
BACKEND_IMAGE="${REGISTRY}/xagent-backend"
FRONTEND_IMAGE="${REGISTRY}/xagent-frontend"
TAG="${1:-latest}"

echo "Building and pushing images with tag: ${TAG}"

# Build backend
echo "Building backend image..."
docker build -f docker/Dockerfile.backend -t "${BACKEND_IMAGE}:${TAG}" .

# Build frontend
echo "Building frontend image..."
docker build -f docker/Dockerfile.frontend -t "${FRONTEND_IMAGE}:${TAG}" ../frontend

# Push images
echo "Pushing images to Docker Hub..."
docker push "${BACKEND_IMAGE}:${TAG}"
docker push "${FRONTEND_IMAGE}:${TAG}"

echo "Images published successfully:"
echo "  - ${BACKEND_IMAGE}:${TAG}"
echo "  - ${FRONTEND_IMAGE}:${TAG}"
