#!/usr/bin/env bash
set -euo pipefail

DOCKER_REPO="junius/cogbase-demo"
VERSION="${1:-latest}"
LOCAL_DATA_DIR="${2:-}"

echo "Pulling $DOCKER_REPO:$VERSION"
docker pull "$DOCKER_REPO:$VERSION"

MOUNT_ARGS=()
if [[ -n "$LOCAL_DATA_DIR" ]]; then
  MOUNT_ARGS=(-v "$LOCAL_DATA_DIR:/data")
fi

echo "Starting cogbase-demo container"
docker run -d \
  --name cogbase-demo \
  -p 8000:8000 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  ${MOUNT_ARGS[@]+"${MOUNT_ARGS[@]}"} \
  "$DOCKER_REPO:$VERSION"

echo "Container started: http://localhost:8000"
