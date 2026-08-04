import os

# Must be set before any test imports main, since module-level code reads
# these via os.getenv at import time (API_KEY gates every mutating route).
os.environ.setdefault("API_KEY", "test-api-key")
os.environ.setdefault("HF_TOKEN", "test-hf-token")
