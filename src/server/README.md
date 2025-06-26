# Centralized Server

## Installation

1. Install dependencies using uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies

```bash
uv sync --locked --no-dev
```
3. Start a consumer (Can be started in a separate terminal, each one will assigned with a different consumer group)
```bash
cd Workspace/reid-microservices/src/server && source .venv/bin/activate && python -m src
```
