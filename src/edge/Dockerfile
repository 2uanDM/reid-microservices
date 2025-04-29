FROM python:3.12.2

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --upgrade pip \
 && pip install uv \
 && uv pip install -r <(uv pip compile pyproject.toml)

CMD ["python", "main.py"]