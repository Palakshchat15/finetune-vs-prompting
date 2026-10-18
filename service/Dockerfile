# DistilBERT Banking77 intent service, CPU-only (no CUDA libraries -> a much smaller image).
# Build from the PROJECT ROOT (the context needs src/, service/ and models/distilbert/):
#   docker build -f service/Dockerfile -t banking77-distilbert:0.1 .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MODEL_DIR=/app/models/distilbert \
    TORCH_NUM_THREADS=1

WORKDIR /app

# 1) CPU-only torch from the PyTorch CPU index (about 200 MB wheel instead of several GB with CUDA).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
# 2) The rest (fastapi, uvicorn, transformers, ...), pinned.
COPY requirements-service.txt .
RUN pip install -r requirements-service.txt

# 3) Code (only what the service imports) and the trained checkpoint (baked in: no volume needed on
#    Docker Desktop's single-node cluster).
COPY src/__init__.py src/distilbert.py src/
COPY service/__init__.py service/core.py service/app.py service/
COPY models/distilbert/ models/distilbert/

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER 10001
EXPOSE 8000
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
