# Streaming Service

This is a streaming service that streams video from a Kafka topic to a WebSocket connection.

## Installation

```bash
uv sync
```

## Running the service

```bash
python -m src
```

## Running the service with Docker

```bash
docker build -t streaming-service .
docker run -p 8765:8765 --name thesis-streaming --restart unless-stopped -d --env-file .env --network kafka_kafka-net thesis-streaming
```
