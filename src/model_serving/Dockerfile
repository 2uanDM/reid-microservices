FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends wget gcc ffmpeg libgl1 libglib2.0-0 libpython3-dev

# Project environment variables
ENV UV_CACHE_DIR=/root/.cache/uv UV_COMPILE_BYTECODE=0

# Project initialization
RUN --mount=from=ghcr.io/astral-sh/uv,source=/uv,target=/bin/uv \
    --mount=type=cache,id=thesis-model-serving,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv export --no-hashes | uv pip sync --system -

# Expose port
EXPOSE 8000 8265

# Clean up unnecessary files
RUN apt-get purge -y --auto-remove && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/* /root/.cache/pip /root/.cache/uv

# Copy Python code to the Docker image
COPY . /app

# Run the model serving service
CMD ["serve", "run", "config.yaml"]