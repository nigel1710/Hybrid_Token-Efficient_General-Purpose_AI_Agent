#!/usr/bin/env bash
set -euo pipefail

# IMPORTANT: Always build with --platform linux/amd64
# Building on Apple Silicon without this flag produces an incompatible image that scores zero.

REGISTRY="${REGISTRY:-docker.io}"
TEAM_NAME="${TEAM_NAME:-your-team-name}"
TAG="${TAG:-latest}"
IMAGE="${REGISTRY}/${TEAM_NAME}/track1-agent:${TAG}"

echo "Building image: $IMAGE (linux/amd64)"

# Step 1: Ensure buildx builder is active
docker buildx create --use --name amd64-builder 2>/dev/null || docker buildx use amd64-builder

# Step 2: Build and push
docker buildx build \
  --platform linux/amd64 \
  --tag "$IMAGE" \
  --push \
  .

echo "Push complete: $IMAGE"

# Step 3: Confirm image size
docker pull "$IMAGE"
docker images "$IMAGE"

echo ""
echo "CHECKLIST:"
echo "  [ ] Image size is under 10GB"
echo "  [ ] Registry repository is set to PUBLIC visibility"
echo "  [ ] Tested locally with real FIREWORKS_* env vars"
echo "  [ ] No .env file inside the image (verify: docker run --rm $IMAGE ls -la /app)"
