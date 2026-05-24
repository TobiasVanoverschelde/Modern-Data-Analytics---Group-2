# syntax=docker/dockerfile:1.7

# Prerequisite: run `python -m src.pipeline && python -m src.training` once
# before building, so data/processed/ contains the parquet artifacts and the
# trained model. The runtime stage copies these directly from the build context
# instead of regenerating them inside the image.
#
# build: docker build -t mda-cycling .
# run:   docker run --rm -p 8000:8000 mda-cycling

# Stage 1: install deps with uv
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements-prod.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-cache --compile-bytecode -r requirements-prod.txt


# Stage 2: serve the app with pre-baked artifacts from the local training run
FROM python:3.12-slim AS runtime
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1

# Pre-baked model + features (produced locally by src/pipeline.py + src/training.py)
COPY data/processed/ ./data/processed/

COPY src/ ./src/
COPY app.py .

EXPOSE 8000
RUN useradd app && chown -R app /app
USER app
CMD ["shiny", "run", "--host", "0.0.0.0", "--port", "8000", "app.py"]
