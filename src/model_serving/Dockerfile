FROM python:3.11-slim-buster

WORKDIR /app

RUN pip instal uv \
    && uv sync

COPY . .

CMD ["serve run main:feature_extraction_app"]
