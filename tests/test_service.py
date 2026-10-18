"""FastAPI service, Kubernetes manifests, Dockerfile and load-test helpers. The model is always a fake:
no torch, no trained checkpoint, no Docker. Route tests run only if fastapi is installed."""
import importlib.util

import pytest
import yaml

from service.core import ModelHolder, Settings
from src.config import ROOT


class FakeClassifier:
    labels = ["card_arrival", "top_up_failed", "pin_blocked"]

    def predict(self, texts, top_k=1):
        return [{"label": "card_arrival", "confidence": 0.9,
                 "top_k": [{"label": lab, "prob": p} for lab, p in zip(self.labels, [0.9, 0.07, 0.03])][:top_k]}
                for _ in texts]


def fake_loader(_settings):
    return FakeClassifier()


def failing_loader(_settings):
    raise FileNotFoundError("no checkpoint")


def test_settings_from_env():
    s = Settings.from_env({"MODEL_DIR": "/m", "MAX_LEN": "32", "TORCH_NUM_THREADS": "2"})
    assert (s.model_dir, s.max_len, s.torch_threads) == ("/m", 32, 2)
    assert Settings.from_env({}).max_len == 64


def test_holder_lifecycle_and_predict():
    h = ModelHolder(Settings(), fake_loader)
    assert h.state == "loading" and not h.ready
    with pytest.raises(RuntimeError):
        h.predict("hi")
    h.load_in_background().join(5)
    assert h.ready
    r = h.predict("  where is my card  ", top_k=2)
    assert r["label"] == "card_arrival" and len(r["top_k"]) == 2 and r["latency_ms"] >= 0
    assert h.status()["requests"] == 1


def test_holder_load_failure_is_reported():
    h = ModelHolder(Settings(), failing_loader)
    h.load()
    assert h.state == "failed" and "no checkpoint" in h.error and not h.ready


def test_validation():
    h = ModelHolder(Settings(max_text_chars=10, max_top_k=5), fake_loader)
    assert h.validate("   ", 1) and h.validate("x" * 11, 1) and h.validate("ok", 6)
    assert h.validate("ok", 3) is None


def _client(loader):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from service.app import create_app
    return TestClient(create_app(Settings(), loader, background=False))


def test_routes_ready_service():
    with _client(fake_loader) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/ready").json()["status"] == "ready"
        r = c.post("/predict", json={"text": "my card has not arrived", "top_k": 3})
        assert r.status_code == 200 and r.json()["label"] == "card_arrival" and len(r.json()["top_k"]) == 3
        assert c.post("/predict", json={"text": ""}).status_code == 422
        assert c.post("/predict", json={"text": "x", "top_k": 99}).status_code == 422
        assert c.post("/predict", json={}).status_code == 422


def test_routes_not_ready_service():
    with _client(failing_loader) as c:
        assert c.get("/health").status_code == 200       # liveness stays green: no restart loop
        assert c.get("/ready").status_code == 503         # readiness red: no traffic
        assert c.post("/predict", json={"text": "hi"}).status_code == 503


# ---- Kubernetes manifests and Dockerfile -------------------------------------------------------------
def _docs():
    out = {}
    for p in (ROOT / "k8s").glob("*.yaml"):
        for d in yaml.safe_load_all(p.read_text(encoding="utf-8")):
            if d:
                out[d["kind"]] = d
    return out


def _mem_mib(v: str) -> float:
    return float(v[:-2]) * (1024 if v.endswith("Gi") else 1) if v.endswith(("Mi", "Gi")) else float(v) / 2**20


def test_k8s_manifests_complete_and_fit_the_laptop():
    d = _docs()
    assert {"Namespace", "ConfigMap", "Deployment", "Service", "HorizontalPodAutoscaler", "Kustomization"} <= set(d)
    c = d["Deployment"]["spec"]["template"]["spec"]["containers"][0]
    assert c["readinessProbe"]["httpGet"]["path"] == "/ready" and c["livenessProbe"]["httpGet"]["path"] == "/health"
    assert c["imagePullPolicy"] in ("IfNotPresent", "Never")
    assert c["envFrom"][0]["configMapRef"]["name"] == d["ConfigMap"]["metadata"]["name"]
    lim, req = c["resources"]["limits"], c["resources"]["requests"]
    hpa = d["HorizontalPodAutoscaler"]["spec"]
    assert hpa["minReplicas"] >= 1 and hpa["scaleTargetRef"]["name"] == d["Deployment"]["metadata"]["name"]
    # Docker Desktop is capped at 4 GB: all replicas at their memory limit must stay well under it.
    assert hpa["maxReplicas"] * _mem_mib(lim["memory"]) <= 3 * 1024
    assert _mem_mib(req["memory"]) <= _mem_mib(lim["memory"])
    assert d["ConfigMap"]["data"]["MAX_LEN"] == str(load_max_len())


def load_max_len():
    from src.config import load_config
    return load_config()["distilbert"]["max_len"]


def test_dockerfile_uses_cpu_torch_and_copies_only_needed_files():
    df = (ROOT / "service" / "Dockerfile").read_text(encoding="utf-8")
    assert "download.pytorch.org/whl/cpu" in df and "models/distilbert" in df
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert ignore[[i for i, x in enumerate(ignore) if x and not x.startswith("#")][0]] == "*"


# ---- load-test helpers ------------------------------------------------------------------------------
def _load_test_module():
    spec = importlib.util.spec_from_file_location("load_test", ROOT / "scripts" / "load_test.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_load_test_summary():
    m = _load_test_module()
    p = m.percentiles([float(x) for x in range(1, 101)])
    assert p["p50_ms"] == 50 and p["p95_ms"] == 95 and p["p99_ms"] == 99 and p["max_ms"] == 100
    res = [{"ok": True, "ms": 10.0, "pred": "a", "gold": "a", "server_ms": 5.0},
           {"ok": True, "ms": 20.0, "pred": "b", "gold": "a", "server_ms": 6.0},
           {"ok": False, "ms": 1.0, "error": "HTTP 503"}]
    s = m.summarize_run(res, 2.0, 2)
    assert s["ok"] == 2 and s["errors"] == 1 and s["throughput_rps"] == 1.0 and s["accuracy_of_answers"] == 0.5
    assert s["error_samples"] == ["HTTP 503"]
