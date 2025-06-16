# Edge Device

This is the module to deploy on the edge device.

## Installation

1. Install dependencies using uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies

```bash
uv sync --locked --no-dev
```

3. Optional: Build docker image

```bash
docker build -t thesis-edge .
```

## Run the edge device

1. Run directly

```bash
python -m src --source="non overlap.mp4" --kafka_bootstrap_servers="217.15.165.221:9092,217.15.165.221:9093,217.15.165.221:9094" --model_path weights/best.onnx
```

Arguments:

- `source`: Path to the video file
- `kafka_bootstrap_servers`: Kafka bootstrap servers url, here we use a cluster of 3 kafka servers
- `model_path`: Path to the YOLO model. Default is `weights/best.onnx`
- `reid_topic`: Kafka topic for re-identification. Default is `reid_input`
- `device_id`: Device ID. Default is `edge_device_<random_uuid>`

2. Run using docker

Example 1: Run with a video file mounted from host to container, and use internal hostnames for kafka brokers with resource limits, and using onnx model
```bash
docker run --rm -it --network kafka_kafka-net \
--cpus="4" --memory="512m" \
--mount type=bind,source=/mnt/e/workspace/Dataset/thesis/turn2/edtech_bacony_low_1.mp4,target=/app/non_overlap.mp4 \
thesis-edge \
--source=/app/non_overlap.mp4 \
--kafka_bootstrap_servers="kafka1:29092,kafka2:29093,kafka3:29094" \
--device_id edge_device_1
```

Example 2: Run with a video file mounted from host to container, and use internal hostnames for kafka brokers with resource limits, and using pt model

```bash
docker run --rm -it --network kafka_kafka-net \
--cpus="4" --memory="512m" \
--mount type=bind,source=/mnt/e/workspace/Dataset/thesis/turn2/edtech_bacony_low_1.mp4,target=/app/non_overlap.mp4 \
thesis-edge \
--source=/app/non_overlap.mp4 \
--kafka_bootstrap_servers="kafka1:29092,kafka2:29093,kafka3:29094" \
--device_id edge_device_1 \
--model_path weights/best.pt
```

Resource limits explanation:
- `--cpus="2"`: Limits the container to use at most 2 CPU cores
- `--memory="4g"`: Limits the container to use at most 4GB of RAM

You can adjust these values based on your system resources:
- CPU: Use values like "1", "1.5", "2", "4" etc.
- Memory: Use values like "1g", "2g", "4g", "8g" or "512m", "1024m" etc.

## Benchmark

Here, to compare the FLOPs between CPU-based device, Jetson Nano and GPU, we can use the `benchmark` directory.

1. Build the benchmark

```bash
cd src/edge/benchmark
docker build -t intel-flops-benchmark .
```

2. Run the benchmark with resource limits

```bash
docker run --rm -it --cpus="4" --memory="512m" intel-flops-benchmark
```