# Serving image: bakes in the trained models from ./models/<subset>/.
# Train first (python -m pdm.train --subset all), then: docker build -t pdm-api .
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PDM_MODEL_DIR=/app/models \
    PDM_PREDICTION_LOG=/app/logs/predictions.jsonl

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY pdm/ pdm/
# reference.csv (drift baseline) is dockerignored; monitoring runs outside the container.
COPY models/ models/

RUN useradd --create-home appuser && mkdir -p logs && chown appuser logs
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "pdm.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
