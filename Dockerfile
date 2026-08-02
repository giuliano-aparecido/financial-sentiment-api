FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000

# --proxy-headers/--forwarded-allow-ips: Render sits the app behind a proxy,
# so request.client.host is otherwise the proxy's own address for every
# request, not the real client - which would make IP-based rate limiting
# bucket all traffic together instead of per-client.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000", "--proxy-headers", "--forwarded-allow-ips=*"]
