#!/usr/bin/env bash
set -euo pipefail

DOCKER_REPO="junius/cogbase-demo"
CONTAINER_NAME="cogbase-demo"

usage() {
  cat <<EOF
Usage: $0 <command> [options]

Commands:
  build               Build and tag the image with the current hatch version + :latest
  push                Push the versioned tag and :latest to Docker Hub
  release             build + push in one step
  pull [VERSION]      Pull VERSION from Docker Hub (default: latest)
  run [VERSION] [DIR] Start the container from a local image (use pull first for hub images)
                        DIR  optional host path mounted to /data
  stop                Stop and remove the running container
  logs                Tail logs from the running container

EOF
}

cmd_build() {
  VERSION=$(hatch version)
  echo "Building $DOCKER_REPO:$VERSION"
  docker build \
    -f server/Dockerfile.demo \
    -t "$DOCKER_REPO:$VERSION" \
    -t "$DOCKER_REPO:latest" \
    .
  echo "Built $DOCKER_REPO:$VERSION and :latest"
}

cmd_push() {
  VERSION=$(hatch version)
  echo "Pushing $DOCKER_REPO:$VERSION and :latest"
  docker push "$DOCKER_REPO:$VERSION"
  docker push "$DOCKER_REPO:latest"
  echo "Done: $DOCKER_REPO:$VERSION"
}

cmd_pull() {
  VERSION="${1:-latest}"
  echo "Pulling $DOCKER_REPO:$VERSION"
  docker pull "$DOCKER_REPO:$VERSION"
}

cmd_run() {
  VERSION="${1:-latest}"
  LOCAL_DATA_DIR="${2:-}"

  MOUNT_ARG=""
  if [[ -n "$LOCAL_DATA_DIR" ]]; then
    MOUNT_ARG="-v $LOCAL_DATA_DIR:/data"
  fi

  echo "Starting $CONTAINER_NAME"
  # shellcheck disable=SC2086
  docker run -d \
    --name "$CONTAINER_NAME" \
    -p 8000:8000 \
    $MOUNT_ARG \
    "$DOCKER_REPO:$VERSION"

  echo "Container started: http://localhost:8000"
}

cmd_stop() {
  echo "Stopping $CONTAINER_NAME"
  docker rm -f "$CONTAINER_NAME"
}

cmd_logs() {
  docker logs -f "$CONTAINER_NAME"
}

case "${1:-}" in
  build)   cmd_build ;;
  push)    cmd_push ;;
  release) cmd_build && cmd_push ;;
  pull)    shift; cmd_pull "$@" ;;
  run)     shift; cmd_run "$@" ;;
  stop)    cmd_stop ;;
  logs)    cmd_logs ;;
  *)       usage; exit 1 ;;
esac
