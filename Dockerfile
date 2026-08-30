# syntax=docker/dockerfile:1
# Production image for the binoculars-eu API (PRD §9bis.5, §11.5 ter).
# Multi-stage: builder installs the venv, runtime copies only the result.
#
#   docker build -t binoculars-eu:0.1.0 .
#   docker run --gpus all -p 8000:8000 \
#     -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
#     -e BINOCULARS_EU_DEFAULT_PROFILE=fr \
#     binoculars-eu:0.1.0
#
# Mounting the host HF cache avoids re-downloading ~5 GB of weights on every
# image rebuild and enables offline starts.

FROM python:3.12-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml README.md ./
COPY binoculars_eu ./binoculars_eu

# PRD §8: CUDA wheels for torch, then the package with its [api] extra.
# For a CPU-only image, swap the index for https://download.pytorch.org/whl/cpu.
RUN uv venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python \
        torch --index-url https://download.pytorch.org/whl/cu124 \
    && uv pip install --python /opt/venv/bin/python ".[api]"

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME="/root/.cache/huggingface"
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/binoculars_eu /opt/venv/lib/python3.12/site-packages/binoculars_eu
EXPOSE 8000
CMD ["uvicorn", "binoculars_eu.api:app", "--host", "0.0.0.0", "--port", "8000"]
