# slim image for smaller size
FROM python:3.12-slim

WORKDIR /app

COPY requirements-prod.txt .
RUN pip install -r requirements-prod.txt

COPY src/ ./src/
COPY app.py .
COPY data/processed/ ./data/processed/

EXPOSE 8000

CMD ["shiny", "run", "--host", "0.0.0.0", "--port", "8000", "app.py"]