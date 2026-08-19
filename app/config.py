import os

from dotenv import load_dotenv

# Loads .env into the real process environment if one exists (local dev) -
# a no-op if it doesn't (Render, CI), since real env vars are already set
# there directly. Must run before any os.getenv() call below. This repo
# had no .env-loading mechanism at all until the research-scan DB layer
# needed one - every var below was previously only ever set as a real
# shell/host env var.
load_dotenv()

API_KEY = os.getenv("API_KEY")
HF_API_TOKEN = os.getenv("HF_TOKEN")

# Neon Postgres connection string for the research-scan persistence layer
# (app/db, app/services/scan_persistence.py) - same variable name already
# set on Render (see that service's env vars), not the DATABASE_URL name
# portfolio-manager-backend uses, to avoid requiring a Render-side rename
# of something already configured.
DATABASE_CONNECTION_STRING = os.getenv("DATABASE_CONNECTION_STRING")

MODEL_ARCHITECTURE = os.getenv("MODEL_ARCHITECTURE", "apertus").lower()

DEFAULT_MODELS = {
    "llama": "gaparecido/llama-3.2-3b-financial-reasoner",
    "apertus": "gaparecido/apertus-8b-financial-reasoner",
    "qwen": "gaparecido/qwen-2.5-7b-financial-reasoner",
    "mistral": "gaparecido/mistral-7b-financial-reasoner",
}

HF_MODEL_REPO = os.getenv("HF_MODEL_URL", DEFAULT_MODELS.get(MODEL_ARCHITECTURE, DEFAULT_MODELS["apertus"]))

DEFAULT_HF_INFERENCE_URL = os.getenv("HF_INFERENCE_URL") or f"https://api-inference.huggingface.co/models/{HF_MODEL_REPO}"

ALLOWED_INFERENCE_HOST_SUFFIXES = tuple(
    suffix.strip()
    for suffix in os.getenv(
        "ALLOWED_INFERENCE_HOST_SUFFIXES",
        "ngrok-free.app,ngrok-free.dev,ngrok-free.pizza,ngrok.io,ngrok.app,huggingface.cloud,huggingface.co,modal.run",
    ).split(",")
    if suffix.strip()
)


def is_allowed_inference_host(host: str) -> bool:
    # A plain str.endswith(suffix) has no label boundary, so
    # "evil-huggingface.co" would satisfy suffix "huggingface.co". Requiring
    # an exact match or a "." right before the suffix closes that gap.
    return any(host == suffix or host.endswith("." + suffix) for suffix in ALLOWED_INFERENCE_HOST_SUFFIXES)
