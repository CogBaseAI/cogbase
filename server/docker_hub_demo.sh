#!/usr/bin/env bash
set -euo pipefail

DOCKER_REPO="junius/cogbase-demo"

VERSION=$(hatch version)
echo "Building $DOCKER_REPO:$VERSION"

docker build \
  -f server/Dockerfile.demo \
  -t "$DOCKER_REPO:$VERSION" \
  -t "$DOCKER_REPO:latest" \
  .

echo "Pushing $DOCKER_REPO:$VERSION and :latest"
docker push "$DOCKER_REPO:$VERSION"
docker push "$DOCKER_REPO:latest"

echo "Done: $DOCKER_REPO:$VERSION"
