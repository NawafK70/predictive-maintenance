# Serving image for the RUL API. Bakes in the trained models from ./models/<subset>/.
#   rul-train --subset all
#   docker build -t turbofan-rul .
#   docker run -p 8000:8000 turbofan-rul
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RUL_MODEL_DIR=/app/models \
    RUL_PREDICTION_LOG=/app/logs/predictions.jsonl

WORKDIR /app

# Install dependencies first (cached layer), then the package itself.
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir .

# reference.csv (drift baseline) is dockerignored: monitoring runs outside the container.
COPY models/ models/

RUN useradd --create-home appuser && mkdir -p logs && chown appuser logs
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "turbofan_rul.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
