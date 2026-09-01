FROM python:3.11-slim

# ffmpeg/libgl are needed by opencv once the CV modules land
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
