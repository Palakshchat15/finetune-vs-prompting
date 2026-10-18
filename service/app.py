r"""FastAPI service for the (j) DistilBERT Banking77 intent classifier (CPU inference).

Endpoints
  POST /predict  {"text": "...", "top_k": 3} -> {"label", "confidence", "top_k": [...], "latency_ms", "model"}
  GET  /health   liveness: the process is up (200 even while the model is still loading)
  GET  /ready    readiness: 200 once the model is loaded, else 503 (Kubernetes sends no traffic until then)

Configuration (environment variables; the Kubernetes ConfigMap sets them): MODEL_DIR, MAX_LEN,
TORCH_NUM_THREADS, MAX_TEXT_CHARS, MAX_TOP_K.

Run locally (project venv with requirements-torch.txt + requirements-service.txt):
  $env:MODEL_DIR="models/distilbert"; .venv\Scripts\uvicorn service.app:app --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from service.core import ModelHolder, Settings, default_loader


class PredictRequest(BaseModel):
    text: str = Field(..., description="customer-support message", examples=["My card still hasn't arrived"])
    top_k: int = Field(3, description="number of most likely intents to return")


class Intent(BaseModel):
    label: str
    prob: float


class PredictResponse(BaseModel):
    label: str
    confidence: float
    top_k: list[Intent]
    latency_ms: float
    model: str


def create_app(settings: Settings | None = None, loader=default_loader, background: bool = True) -> FastAPI:
    settings = settings or Settings.from_env()
    holder = ModelHolder(settings, loader)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if background:
            holder.load_in_background()
        else:
            holder.load()
        yield

    app = FastAPI(title="Banking77 intent classifier (DistilBERT)", version="0.1.0", lifespan=lifespan)
    app.state.holder = holder

    @app.get("/health")
    def health():
        return {"status": "ok", **holder.status()}

    @app.get("/ready")
    def ready():
        if not holder.ready:
            raise HTTPException(status_code=503, detail=holder.status())
        return {"status": "ready", **holder.status()}

    @app.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest):
        if not holder.ready:
            raise HTTPException(status_code=503, detail=f"model not ready ({holder.state})")
        err = holder.validate(req.text, req.top_k)
        if err:
            raise HTTPException(status_code=422, detail=err)
        r = await run_in_threadpool(holder.predict, req.text, req.top_k)  # keep the event loop free
        return {**r, "model": settings.model_dir}

    return app


app = create_app()
