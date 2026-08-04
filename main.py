import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from app.config import FRONTEND_ORIGIN
from app.routers import admin, analyze, health
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

# Applies to every route via default_limits, no per-route decorators needed.
# Keyed by client IP - see the Dockerfile's --proxy-headers flag, without
# which every request behind Render's proxy would share one IP and thus one
# bucket. The real cost here is per-request HF inference + yfinance calls,
# so the limit is deliberately tight - this endpoint is not meant for bursts.
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
# Added before CORSMiddleware so CORS ends up as the outer layer (Starlette
# wraps middleware in reverse order of addition) - otherwise a 429 response
# would be missing CORS headers and the browser would see an opaque network
# error instead of a readable 429.
app.add_middleware(SlowAPIMiddleware)

# ==============================================================================
# ALLOW CORS FOR VERCEL FRONTEND REQUESTS
# ==============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type", "X-API-Key"],
)

app.include_router(health.router)
app.include_router(analyze.router)
app.include_router(admin.router)
