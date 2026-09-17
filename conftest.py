import os

import pytest
from fastapi.testclient import TestClient

# Must be set before any test imports main, since module-level code reads
# these via os.getenv at import time (API_KEY gates every mutating route).
os.environ.setdefault("API_KEY", "test-api-key")
os.environ.setdefault("HF_TOKEN", "test-hf-token")
os.environ.setdefault("DEFAULT_MODEL", "llama")
# Every test using the `client` fixture below runs the app's REAL lifespan
# (TestClient as a context manager) - without this, app/services/
# scheduler.py's catch-up-on-stale logic would hit the real Neon DB and
# schedule a genuine live Yahoo-scanning job on every such test. See that
# module's start() docstring for the incident this guards against.
os.environ.setdefault("RESEARCH_SCHEDULER_DISABLED", "1")

from main import app  # noqa: E402
from app.services import inference  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_model_registry(monkeypatch):
    monkeypatch.setattr(inference, "_inference_urls", dict(inference._inference_urls))


@pytest.fixture
def client():
    # Entering as a context manager runs the app's lifespan, which starts
    # the shared httpx client that /api/analyze depends on.
    with TestClient(app) as c:
        yield c
