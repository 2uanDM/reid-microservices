# Command to run the edge device

```bash
python main.py --source="non overlap.mp4" --kafka_server_uri="<SERVER_IP>:9092,<SERVER_IP>:9093,<SERVER_IP>:9094" --model_path weights/best.onnx
```