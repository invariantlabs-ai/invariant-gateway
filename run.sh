up() {
  # Ensure the main network exists
  docker network inspect invariant-explorer-web >/dev/null 2>&1 || \
    docker network create invariant-explorer-web

  # Start your local docker-compose services
  docker compose -f docker-compose.local.yml up -d

  echo "Proxy started at http://localhost/api/v1/proxy/"
  echo "See http://localhost/api/v1/proxy/docs for API documentation"
}

build() {
  # Build local services
  docker compose -f docker-compose.local.yml build
}

down() {
  # Bring down local services
  docker compose -f docker-compose.local.yml down
}


tests() {
    # Run tests
    pip install invariant-ai
    invariant explorer up -d --build

    until curl -X GET -I http://127.0.0.1/api/v1 --fail --silent --output /dev/null; do
      echo "Backend API not available yet - checking health..."
      sleep 2
    done

    echo "Backend API is available. Running tests..."

    docker build -t 'explorer-proxy-test' -f ./tests/Dockerfile.test ./tests

    docker run \
    --mount type=bind,source=./tests,target=/tests \
    --network host \
    explorer-proxy-test $@
}

# -----------------------------
# Command dispatcher
# -----------------------------
case "$1" in
  "up")
    up
    ;;
  "build")
    build
    ;;
  "down")
    down
    ;;
  "logs")
    docker compose -f docker-compose.local.yml logs -f
    ;;
  "tests")
    shift
    tests $@
    ;;
  *)
    echo "Usage: $0 [up|build|down|logs|tests]"
    exit 1
    ;;
esac