# Command to run the edge device

```bash
python main.py --source="non overlap.mp4" --kafka_bootstrap_servers="217.15.165.221:9092,217.15.165.221:9093,217.15.165.221:9094" --model_path weights/best.onnx
```

```bash
python main.py --source="non overlap.mp4" --kafka_bootstrap_servers="localhost:9092,localhost:9093,localhost:9094" --model_path weights/best.onnx
```