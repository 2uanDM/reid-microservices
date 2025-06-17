# Kafka message broker cluster

This `docker-compose.yaml` file include the script to start the Kafka message broker cluster with one Zookeeper and three Kafka brokers.

Note: Need `.env` file with the following variables set:
- `KAFKA_PUBLIC_IP`: For example, if your edge devices / consumers will connect to the Kafka cluster using the public IP of the host machine, set this variable to that IP address.

Similarly, if you just want to run the Kafka cluster locally, you can set this variable to `localhost`.

Or when running inside Docker, and the edge devices also run inside Docker, this variable does not required, since the Kafka brokers will be accessible by their container names as the internal DNS resolution `kafka1`, `kafka2`, `kafka3`.

Command:
```bash
docker compose up -d
```