import os
import re
from collections.abc import Mapping

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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"


def require_hf_api_token() -> None:
    """Fails loudly and specifically if HF_API_TOKEN is missing - same
    style as DATABASE_CONNECTION_STRING's check in app/db/session.py's
    _session_factory(). Without this, app/services/inference.py's
    _call_model sends `Authorization: Bearer None` on every call, and
    every one fails with the SAME generic 502 a real backend outage would
    produce - indistinguishable from this repo's own misconfiguration.
    Called from inference.start_client() (app startup, main.py's
    lifespan), not at import time here, so unrelated code that merely
    imports this module (alembic migrations, tests that never exercise
    inference) isn't forced to have HF_TOKEN set just to import it - same
    lazy-check reasoning as _session_factory()'s own comment."""
    if not HF_API_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is not set - required for calling the Hugging Face inference backend (app/services/inference.py)."
        )


# Neon Postgres connection string for the research-scan persistence layer
# (app/db, app/services/scan_persistence.py) - same variable name already
# set on Render (see that service's env vars), not the DATABASE_URL name
# portfolio-manager-backend uses, to avoid requiring a Render-side rename
# of something already configured.
DATABASE_CONNECTION_STRING = os.getenv("DATABASE_CONNECTION_STRING")

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM") or SMTP_USER


def _parse_email_list(raw: str) -> tuple[str, ...]:
    return tuple(address.strip() for address in raw.split(",") if address.strip())


ALERT_EMAIL_TO = _parse_email_list(os.getenv("ALERT_EMAIL_TO", ""))

MODEL_NAME_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")

_INFERENCE_URL_SUFFIX = "_INFERENCE_URL"


def normalize_model_name(name: str) -> str:
    normalized = name.strip().lower()
    if not MODEL_NAME_RE.fullmatch(normalized):
        raise ValueError(
            f"Invalid model name {name!r}: must start with a letter or digit, then letters, digits, '.', '-' or '_' (max 64 chars)"
        )
    return normalized


def inference_urls_from_env(env: Mapping[str, str]) -> dict[str, str]:
    """Every `<NAME>_INFERENCE_URL` env var registers model `<name>`."""
    urls: dict[str, str] = {}
    for var, url in env.items():
        if not var.endswith(_INFERENCE_URL_SUFFIX) or not url.strip():
            continue
        urls[normalize_model_name(var[: -len(_INFERENCE_URL_SUFFIX)])] = url.strip()
    return urls


DEFAULT_MODEL = normalize_model_name(os.getenv("DEFAULT_MODEL") or "llama")

INFERENCE_URLS = inference_urls_from_env(os.environ)
if DEFAULT_MODEL not in INFERENCE_URLS:
    raise RuntimeError(f"DEFAULT_MODEL={DEFAULT_MODEL!r} has no endpoint: set {DEFAULT_MODEL.upper()}_INFERENCE_URL")


def _parse_allowed_host_suffixes(raw: str) -> tuple[str, ...]:
    # Lowercased so this matches regardless of the case an operator sets
    # ALLOWED_INFERENCE_HOST_SUFFIXES in - is_allowed_inference_host below
    # compares against a host its caller has already lowercased (see
    # admin.py's update_inference_url), so a mixed-case suffix here used to
    # silently never match anything, fail-closed.
    return tuple(suffix.strip().lower() for suffix in raw.split(",") if suffix.strip())


ALLOWED_INFERENCE_HOST_SUFFIXES = _parse_allowed_host_suffixes(
    os.getenv(
        "ALLOWED_INFERENCE_HOST_SUFFIXES",
        "ngrok-free.app,ngrok-free.dev,ngrok-free.pizza,ngrok.io,ngrok.app,huggingface.cloud,huggingface.co,modal.run",
    )
)


def is_allowed_inference_host(host: str) -> bool:
    # A plain str.endswith(suffix) has no label boundary, so
    # "evil-huggingface.co" would satisfy suffix "huggingface.co". Requiring
    # an exact match or a "." right before the suffix closes that gap.
    return any(host == suffix or host.endswith("." + suffix) for suffix in ALLOWED_INFERENCE_HOST_SUFFIXES)
