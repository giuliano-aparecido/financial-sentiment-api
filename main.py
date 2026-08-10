import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.limiter import limiter
from app.routers import admin, analyze, health, research
from app.services import inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Shared client so every /api/analyze request reuses one connection pool
    # instead of paying TCP/TLS setup on each call.
    await inference.start_client()
    try:
        yield
    finally:
        await inference.stop_client()


app = FastAPI(title="Multi-Model Financial RAG Reasoning Engine", lifespan=lifespan)

# limiter itself now lives in app/limiter.py, so router modules (see
# app/routers/research.py) can import it for per-route overrides without a
# circular import back to this file.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.include_router(health.router)
app.include_router(analyze.router)
app.include_router(admin.router)
app.include_router(research.router)
