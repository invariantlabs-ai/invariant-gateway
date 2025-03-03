up() {
  # Ensure the main network exists
  docker network inspect invariant-explorer-web >/dev/null 2>&1 || \
    docker network create invariant-explorer-web

  # Default values
  POLICIES_FILE_PATH=""

  # Parse command-line arguments
  while [[ "$#" -gt 0 ]]; do
      case "$1" in
          --policies-file=*)
              POLICIES_FILE_PATH="${1#*=}"
              ;;
          *)
              echo "Unknown parameter: $1"
              exit 1
              ;;
      esac
      shift
  done

  if [[ -n "$POLICIES_FILE_PATH" ]]; then
    if [[ -f "$POLICIES_FILE_PATH" ]]; then
      POLICIES_FILE_PATH=$(realpath "$POLICIES_FILE_PATH")
    else
      echo "Error: Specified policies file does not exist: $POLICIES_FILE_PATH"
      exit 1
    fi
  fi

  # Start Docker Compose with the correct environment variable
  POLICIES_FILE_PATH="$POLICIES_FILE_PATH" docker compose -f docker-compose.local.yml up -d

  echo "Proxy started at http://localhost:8005/api/v1/proxy/"
  echo "See http://localhost:8005/api/v1/proxy/docs for API documentation"
  echo "Using Policies File: ${POLICIES_FILE_PATH:-None}"
}

build() {
  # Build local services
  docker compose -f docker-compose.local.yml build
}

down() {
  # Bring down local services
  docker compose -f docker-compose.local.yml down
  docker compose -f tests/docker-compose.test.yml down
}


tests() {
  echo "Setting up test environment..."

  # Ensure test network exists
  docker network inspect invariant-proxy-web-test >/dev/null 2>&1 || \
    docker network create invariant-proxy-web-test

  # Setup the explorer.test.yml file
  CONFIG_DIR="/tmp/invariant-proxy-test/configs"
  FILE="$CONFIG_DIR/explorer.test.yml"
  mkdir -p "$CONFIG_DIR"
  # Download the file
  curl -L -o "$FILE" https://raw.githubusercontent.com/invariantlabs-ai/explorer/main/configs/explorer.test.yml
  # Verify if the file exists
  if [ ! -f "$FILE" ]; then
    echo "Error: File $FILE not found. Issue with download."
    exit 1
  fi
  echo "File successfully downloaded: $FILE"

  # Start containers
  docker compose -f tests/docker-compose.test.yml down
  docker compose -f tests/docker-compose.test.yml build
  docker compose -f tests/docker-compose.test.yml up -d

  until [ "$(docker inspect -f '{{.State.Health.Status}}' invariant-proxy-test-explorer-app-api)" = "healthy" ]; do
    echo "Explorer backend app-api instance container starting..."
    sleep 2
  done

  until [ "$(docker inspect -f '{{.State.Health.Status}}' invariant-proxy-test)" = "healthy" ]; do
    echo "Invariant proxy test instance container starting..."
    sleep 2
  done

  echo "Running tests..."

  # Make call to signup endpoint
  curl -k -X POST http://127.0.0.1/api/v1/user/signup

  docker build -t 'invariant-proxy-tests' -f ./tests/Dockerfile.test ./tests

  docker run \
    --mount type=bind,source=./tests,target=/tests \
    --network invariant-proxy-web-test \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"\
    --env-file ./tests/.env.test \
    invariant-proxy-tests $@
}

# -----------------------------
# Command dispatcher
# -----------------------------
case "$1" in
  "up")
    shift
    up $@
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