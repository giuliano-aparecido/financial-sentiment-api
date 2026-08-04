# Pinned by manifest-list digest (not just the floating "3.11-slim" tag) so
# a base-image update can't silently change what gets deployed.
FROM python:3.11-slim@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# .dockerignore keeps tests/, .git, __pycache__, etc. out of the image.
COPY . .

# Run as an unprivileged user - nothing in this app needs root.
RUN useradd --create-home --shell /usr/sbin/nologin appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:10000/health', timeout=2)" || exit 1

# --proxy-headers/--forwarded-allow-ips: Render sits the app behind a proxy,
# so request.client.host is otherwise the proxy's own address for every
# request. The rate limiter no longer keys on this (see main.py's
# _rate_limit_key) since trusting X-Forwarded-For from any caller made that
# spoofable - IP is now used for logging only (see admin.py), where a
# spoofed value is a minor annoyance, not a security bypass.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000", "--proxy-headers", "--forwarded-allow-ips=*"]
