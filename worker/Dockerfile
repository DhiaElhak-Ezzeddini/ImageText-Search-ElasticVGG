# worker/Dockerfile
FROM python:3.10-slim

# Install system dependencies for opencv, font support and general build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create app directories
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy requirement file early to leverage layer cache
COPY worker/requirements.txt /app/requirements.txt

# Install pip deps
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy code (shared + worker scripts)
COPY shared /app/shared
COPY worker /app/worker

# Ensure Python can import shared as a module
ENV PYTHONPATH=/app

# Default workdir for running commands (indexer/trainer)
WORKDIR /app/worker

# No default CMD so callers can override with trainer or indexer commands
ENTRYPOINT ["python"]
