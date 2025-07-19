lint: ## Run linters
	@echo "🚀 Running Ruff lint"
	@ruff check . --fix

edge1: ## Run edge1
	@echo "🚀 Running edge1"
	@docker run --rm -it --network kafka_kafka-net \
--cpus="1" --memory="512m" \
--mount type=bind,source=/mnt/e/workspace/Dataset/thesis/turn2/demo/device_1.mp4,target=/app/non_overlap.mp4 \
thesis-edge \
--source=/app/non_overlap.mp4 \
--kafka_bootstrap_servers="kafka1:29092,kafka2:29093,kafka3:29094" \
--device_id edge_device_1 \
--model_path weights/yolo11n.pt

edge2: ## Run edge2
	@echo "🚀 Running edge2"
	@docker run --rm -it --network kafka_kafka-net \
--cpus="1" --memory="512m" \
--mount type=bind,source=/mnt/e/workspace/Dataset/thesis/turn2/demo/device_2.mp4,target=/app/non_overlap.mp4 \
thesis-edge \
--source=/app/non_overlap.mp4 \
--kafka_bootstrap_servers="kafka1:29092,kafka2:29093,kafka3:29094" \
--device_id edge_device_2 \
--model_path weights/yolo11n.pt

edge3: ## Run edge3
	@echo "🚀 Running edge3"
	@docker run --rm -it --network kafka_kafka-net \
--cpus="1" --memory="512m" \
--mount type=bind,source=/mnt/e/workspace/Dataset/thesis/turn2/demo/device_3.mp4,target=/app/non_overlap.mp4 \
thesis-edge \
--source=/app/non_overlap.mp4 \
--kafka_bootstrap_servers="kafka1:29092,kafka2:29093,kafka3:29094" \
--device_id edge_device_3 \
--model_path weights/yolo11n.pt

edge:
	@echo "🚀 Running 3 edges"
	@docker run --rm --network kafka_kafka-net --name edge_1 -d \
--cpus="1" --memory="512m" \
--mount type=bind,source=/mnt/e/workspace/Dataset/thesis/turn2/demo/device_1.mp4,target=/app/non_overlap.mp4 \
thesis-edge \
--source=/app/non_overlap.mp4 \
--kafka_bootstrap_servers="kafka1:29092,kafka2:29093,kafka3:29094" \
--device_id edge_device_1 \
--model_path weights/yolo11n.pt && \
docker run --rm --network kafka_kafka-net --name edge_2 -d \
--cpus="1" --memory="512m" \
--mount type=bind,source=/mnt/e/workspace/Dataset/thesis/turn2/demo/device_2.mp4,target=/app/non_overlap.mp4 \
thesis-edge \
--source=/app/non_overlap.mp4 \
--kafka_bootstrap_servers="kafka1:29092,kafka2:29093,kafka3:29094" \
--device_id edge_device_2 \
--model_path weights/yolo11n.pt && \
docker run --rm --network kafka_kafka-net --name edge_3 -d \
--cpus="1" --memory="512m" \
--mount type=bind,source=/mnt/e/workspace/Dataset/thesis/turn2/demo/device_3.mp4,target=/app/non_overlap.mp4 \
thesis-edge \
--source=/app/non_overlap.mp4 \
--kafka_bootstrap_servers="kafka1:29092,kafka2:29093,kafka3:29094" \
--device_id edge_device_3 \
--model_path weights/yolo11n.pt

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help