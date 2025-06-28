# Build stage
FROM python:3.11-slim-bookworm

WORKDIR /app

# Copy dependency files first for better caching
COPY requirements.txt ./

# Install dependencies in virtual environment
RUN pip install -r requirements.txt

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . .

# Remove env file
RUN rm -f .env

# Set the entrypoint to run main.py with arguments
ENTRYPOINT ["python", "-m", "src"]