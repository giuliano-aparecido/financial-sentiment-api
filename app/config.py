import os

API_KEY = os.getenv("API_KEY")
HF_API_TOKEN = os.getenv("HF_TOKEN")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")

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
        "ngrok-free.app,ngrok-free.dev,ngrok-free.pizza,ngrok.io,ngrok.app,huggingface.cloud,huggingface.co",
    ).split(",")
    if suffix.strip()
)
