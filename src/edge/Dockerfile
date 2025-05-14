FROM python:3.11-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --upgrade pip \
 && pip install uv \
 && uv sync
