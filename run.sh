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
esac