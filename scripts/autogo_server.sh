#!/bin/bash
# AutoGo server helper — build + run AutoGo Docker container for head-to-head benchmarking.
#
# Usage:
#   ./scripts/autogo_server.sh          # Start server
#   ./scripts/autogo_server.sh stop     # Stop server
#   ./scripts/autogo_server.sh status   # Check if running
#
# Plan 065 Phase 0 T6: API Bridge prerequisite

set -euo pipefail

CONTAINER_NAME="autogo"
HOST_PORT=8000
IMAGE_NAME="autogo-dev"

# Resolve project root from git
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && git rev-parse --show-toplevel 2>/dev/null || echo "$(dirname "$0")/..")"
AUTOGO_DIR="${PROJECT_ROOT}/.raw/autogo"

# Colors for output (disable if not a terminal)
if [ -t 1 ]; then
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    RED='\033[0;31m'
    NC='\033[0m'
else
    GREEN=''
    YELLOW=''
    RED=''
    NC=''
fi

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Commands ────────────────────────────────────────────────────

cmd_stop() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        info "Stopping ${CONTAINER_NAME} container..."
        docker stop "${CONTAINER_NAME}" 2>/dev/null || true
        docker rm "${CONTAINER_NAME}" 2>/dev/null || true
        info "Container removed."
    else
        warn "No ${CONTAINER_NAME} container found."
    fi
}

cmd_status() {
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        info "AutoGo server is ${GREEN}RUNNING${NC} on http://localhost:${HOST_PORT}"
        echo ""
        curl -sf "http://localhost:${HOST_PORT}/api/agents" | python3 -m json.tool 2>/dev/null \
            || warn "Server running but /api/agents not responding (still starting?)"
    elif docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        warn "AutoGo container exists but is ${RED}STOPPED${NC}"
    else
        warn "No AutoGo container found."
    fi
}

cmd_start() {
    # Check if already running
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        info "AutoGo server already running on http://localhost:${HOST_PORT}"
        cmd_status
        return 0
    fi

    # Check for AutoGo source
    if [ ! -d "${AUTOGO_DIR}" ]; then
        error "AutoGo source not found at: ${AUTOGO_DIR}"
        error "Clone or symlink the AutoGo repo to .raw/autogo/"
        exit 1
    fi

    if [ ! -f "${AUTOGO_DIR}/.devcontainer/Dockerfile" ]; then
        error "Dockerfile not found at: ${AUTOGO_DIR}/.devcontainer/Dockerfile"
        exit 1
    fi

    # Remove old container if exists
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        info "Removing old container..."
        docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    fi

    # Build image — the slim SERVER image (self-contained: alpha_go package
    # + built alpha_go_cpp + gnugo; the devcontainer Dockerfile is a DEV
    # image whose workspace is bind-mounted and whose install runs in
    # postCreateCommand, so it cannot serve from a plain docker run).
    # The Dockerfile is COMMITTED here (scripts/autogo_server.Dockerfile);
    # the four research-local patches it needs in the AutoGo tree are
    # listed in scripts/autogo_server_patches.md.
    info "Building ${IMAGE_NAME} image..."
    docker build -f "${PROJECT_ROOT}/scripts/autogo_server.Dockerfile" -t "${IMAGE_NAME}" "${AUTOGO_DIR}" 2>&1 | tail -5

    # Run container (the image's CMD already starts the server; no
    # override — the image has no uv).
    info "Starting AutoGo server on port ${HOST_PORT}..."
    docker run -d \
        --name "${CONTAINER_NAME}" \
        -p "${HOST_PORT}:8000" \
        "${IMAGE_NAME}"

    # Wait for server to be ready
    info "Waiting for server to start..."
    for i in $(seq 1 30); do
        if curl -sf "http://localhost:${HOST_PORT}/api/agents" >/dev/null 2>&1; then
            echo ""
            info "AutoGo server is ${GREEN}READY${NC} at http://localhost:${HOST_PORT}"
            echo ""
            info "Available agents:"
            curl -sf "http://localhost:${HOST_PORT}/api/agents" | python3 -m json.tool
            echo ""
            info "Run benchmark:"
            info "  cargo run --features go --example go_00_api_bridge"
            return 0
        fi
        printf "  [%d/30] waiting...\r" "$i"
        sleep 1
    done

    echo ""
    error "Server did not respond within 30 seconds."
    error "Check logs: docker logs ${CONTAINER_NAME}"
    exit 1
}

# ── Main ────────────────────────────────────────────────────────

case "${1:-start}" in
    start)  cmd_start  ;;
    stop)   cmd_stop   ;;
    status) cmd_status ;;
    *)
        echo "Usage: $0 {start|stop|status}"
        echo ""
        echo "  start  - Build and start AutoGo server (default)"
        echo "  stop   - Stop and remove container"
        echo "  status - Check if server is running"
        exit 1
        ;;
esac
