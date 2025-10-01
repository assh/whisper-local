# ---- FastAPI runtime ----
FROM python:3.11-slim

WORKDIR /app

# System deps (ffmpeg for faster-whisper, build tools for some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# App code
COPY server.py /app/server.py
# Optional: copy .env files if you want them inside the image (usually better to pass env at runtime)
# COPY .env ./.env
# COPY .env.local ./.env.local

# Environment (override at run-time as needed)
ENV PORT=8000 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Healthcheck so Docker knows it’s healthy
HEALTHCHECK --interval=20s --timeout=3s --retries=5 \
  CMD curl -fsS http://localhost:8000/openapi.json || exit 1

# Start the API
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]