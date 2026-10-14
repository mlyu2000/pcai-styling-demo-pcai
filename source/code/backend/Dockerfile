# PCAI Styling Demo — backend API image (FastAPI orchestrator).
FROM python:3.11-slim

# Build version (the helm tag, e.g. v39) — baked in at build time so the UI can
# show which backend build is actually running.
ARG BUILD_TAG=dev
ENV BUILD_TAG=${BUILD_TAG}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=0

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
# sample_data/ holds the mock assets (sample_scene_dusk.png) so the demo's
# use_mock_scene path works with no customer input and no network fetch.
COPY sample_data/ ./sample_data/

RUN useradd -m -u 1000 app \
    && mkdir -p /results && chown -R app:app /app /results
USER 1000

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
