# syntax=docker/dockerfile:1.7

#build: docker build -t mda-cycling . 
#run: docker run --rm -p 8000:8000 mda-cycling

# Stage 1: install deps with uv
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements-prod.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-cache --compile-bytecode -r requirements-prod.txt


# Stage 2: run the pipeline and train the model
FROM python:3.12-slim AS bootstrap
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1
COPY src/ ./src/
RUN python -m src.pipeline
RUN python -m src.training


# Stage 3: serve the app
FROM python:3.12-slim AS runtime
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1

COPY --from=bootstrap /app/data/processed ./data/processed
COPY --from=bootstrap /app/notebooks/mlruns ./notebooks/mlruns
COPY src/ ./src/
COPY app.py .

EXPOSE 8000
RUN useradd app && chown -R app /app
USER app
CMD ["shiny", "run", "--host", "0.0.0.0", "--port", "8000", "app.py"]
