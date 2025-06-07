# Thesis Model Serving

This is a simple model serving service for the thesis project. It is built using Ray Serve and Docker.

## Building the Docker image

```bash
docker build -t thesis-model-serving .
```

## Running the Docker container

```bash
docker run -p 8000:8000 -p 8265:8265 -d --gpus all --name thesis-serve thesis-model-serving
```