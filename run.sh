up() {
  # Ensure the main network exists
  docker network inspect invariant-explorer-web >/dev/null 2>&1 || \
    docker network create invariant-explorer-web

  # Default values
  GUARDRAILS_FILE_PATH=""

  # Parse command-line arguments
  while [[ "$#" -gt 0 ]]; do
      case "$1" in
          --guardrails-file=*)
              GUARDRAILS_FILE_PATH="${1#*=}"
              ;;
          *)
              echo "Unknown parameter: $1"
              exit 1
              ;;
      esac
      shift
  done

  if [[ -n "$GUARDRAILS_FILE_PATH" ]]; then
    if [[ -f "$GUARDRAILS_FILE_PATH" ]]; then
      GUARDRAILS_FILE_PATH=$(realpath "$GUARDRAILS_FILE_PATH")
      export GUARDRAILS_FILE_PATH="$GUARDRAILS_FILE_PATH"
    else
      echo "Error: Specified guardrails file does not exist: $GUARDRAILS_FILE_PATH"
      exit 1
    fi

    # If GUARDRAILS_FILE_PATH is set, then INVARIANT_API_KEY **must** be set
    if [[ -z "$INVARIANT_API_KEY" ]]; then
      echo "Error: A guardrails file is specified, but INVARIANT_API_KEY env var is not set. This is required to validate guardrails."
      exit 1
    fi
  fi

  # Start Docker Compose with the correct environment variable
  docker compose -f docker-compose.local.yml up -d

  # Get the status of the container
  sleep 2

  if [ -z "$(docker ps -qf 'name=invariant-gateway')" ]; then
    echo "The invariant-gateway container failed to start."
    docker logs invariant-gateway | tail -20  # Show last 20 lines of logs
    exit 1
  fi

  echo "Gateway started at http://localhost:8005/api/v1/gateway/"
  echo "See http://localhost:8005/api/v1/gateway/docs for API documentation"
  if [ -n "$GUARDRAILS_FILE_PATH" ]; then
    echo "Using Guardrails File: $GUARDRAILS_FILE_PATH"
  fi
  unset GUARDRAILS_FILE_PATH
}

build() {
  # Build local services
  docker compose -f docker-compose.local.yml build
}

down() {
  # Bring down local services
  docker compose -f docker-compose.local.yml down
  GATEWAY_ROOT_PATH=$(pwd) docker compose -f tests/integration/docker-compose.test.yml down
}

unit_tests() {
  echo "Running unit tests..."
  PYTHONPATH=. pytest tests/unit_tests $@
}

integration_tests() {
  echo "Setting up test environment to run integration tests..."

  # Ensure test network exists
  docker network inspect invariant-gateway-web-test >/dev/null 2>&1 || \
    docker network create invariant-gateway-web-test

  # Setup the explorer.test.yml file
  CONFIG_DIR="/tmp/invariant-gateway-test/configs"
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

  if [[ -z "$INVARIANT_API_KEY" ]]; then
    echo "Error: INVARIANT_API_KEY env var is not set. This is required to run integration tests."
    exit 1
  fi

  TEST_GUARDRAILS_FILE_PATH="tests/integration/resources/guardrails/integration_test_guardrails_via_file.py"
  if [[ -n "$TEST_GUARDRAILS_FILE_PATH" ]]; then
    if [[ -f "$TEST_GUARDRAILS_FILE_PATH" ]]; then
      TEST_GUARDRAILS_FILE_PATH=$(realpath "$TEST_GUARDRAILS_FILE_PATH")
    else
      echo "Error: Specified test guardrails file does not exist: $TEST_GUARDRAILS_FILE_PATH"
      exit 1
    fi
  fi

  export GATEWAY_ROOT_PATH=$(pwd)
  export GUARDRAILS_FILE_PATH="$TEST_GUARDRAILS_FILE_PATH"

  # Start containers
  docker compose -f tests/integration/docker-compose.test.yml down
  docker compose -f tests/integration/docker-compose.test.yml build
  docker compose -f tests/integration/docker-compose.test.yml up -d

  until [ "$(docker inspect -f '{{.State.Health.Status}}' invariant-gateway-test-explorer-app-api)" = "healthy" ]; do
    echo "Explorer backend app-api instance container starting..."
    sleep 2
  done

  until [ "$(docker inspect -f '{{.State.Health.Status}}' invariant-gateway-test)" = "healthy" ]; do
    echo "Invariant gateway test instance container starting..."
    sleep 2
  done

  while true; do
    echo "Attempting to create test user in invariant-gateway-test-explorer-app-api"
    RESPONSE=$(curl -ks -X POST http://127.0.0.1/api/v1/user/signup)
    if echo "$RESPONSE" | jq -e '.success == true' >/dev/null 2>&1; then
        echo "Created test user in invariant-gateway-test-explorer-app-api"
        break
    fi
    echo "$RESPONSE"
    sleep 2
  done

  # Generate latest whl file for the invariant-gateway package. 
  # This is required to run the integration tests.
  pip install build
  python -m build
  WHEEL_FILE=$(ls dist/*.whl | head -n 1)
  echo "WHEEL_FILE: $WHEEL_FILE"

  echo "Running integration tests..."

  docker build -t 'invariant-gateway-tests' -f ./tests/integration/Dockerfile.test ./tests

  docker run \
    --mount type=bind,source=./tests/integration,target=/tests \
    --mount type=bind,source=$(realpath $WHEEL_FILE),target=/package/$(basename $WHEEL_FILE) \
    --network invariant-gateway-web-test \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"\
    -e GEMINI_API_KEY="$GEMINI_API_KEY" \
    -e INVARIANT_API_KEY="$INVARIANT_API_KEY" \
    --env-file ./tests/integration/.env.test \
    invariant-gateway-tests $@

  unset GATEWAY_ROOT_PATH
  unset GUARDRAILS_FILE_PATH
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
  "unit-tests")
    shift
    unit_tests $@
    ;;
  "integration-tests")
    shift
    integration_tests $@
    ;;
  *)
    echo "Usage: $0 [up|build|down|logs|unit-tests|integration-tests]"
    exit 1
    ;;
esac