# AutoGo REST server image (headless arena arm) - slim, CPU-only.
#
# The repo's .devcontainer/Dockerfile is a DEV image (bind-mount workspace +
# postCreateCommand install), so plain `docker run` has no alpha_go package
# and no built extension. This image is self-contained: it installs the
# MINIMAL server runtime (fastapi/uvicorn/pydantic/numpy - the nn agents'
# torch is optional and absent), builds the alpha_go_cpp extension, and
# ships gnugo for both the tracker engine and the gnugo1 agent.
FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        gnugo cmake build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY src/ src/
COPY scripts/build_cpp.sh scripts/build_cpp.sh

ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src
RUN python -m venv /app/.venv \
    && pip install --no-cache-dir fastapi "uvicorn[standard]" numpy pydantic python-multipart \
    && ./scripts/build_cpp.sh

EXPOSE 8000
CMD ["/app/.venv/bin/python", "-m", "alpha_go.play", "--host", "0.0.0.0", "--port", "8000"]
