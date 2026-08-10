import os

API_KEY = os.getenv("API_KEY")
HF_API_TOKEN = os.getenv("HF_TOKEN")

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
