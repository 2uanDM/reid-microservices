#!/usr/bin/env python3
# chmod +x gen_kafka_compose.py
# ./gen_kafka_compose.py -b 5 -o docker-compose.yaml

import yaml
import argparse

def make_zk_services():
    zk_svcs = {}
    for i in range(1, 4):  # luôn cố định 3 ZooKeeper
        zk_svcs[f'zookeeper{i}'] = {
            'image': 'zookeeper:3.8',
            'container_name': f'zookeeper{i}',
            'hostname': f'zookeeper{i}',
            'ports': [f"{2180+i}:2181"],
            'environment': {
                'ZOO_MY_ID': i,
                'ZOO_SERVERS': "\n".join(
                    [f"server.{j}=zookeeper{j}:2888:3888;2181" for j in range(1, 4)]
                )
            },
            'volumes': [
                f"zookeeper{i}_data:/data",
                f"zookeeper{i}_log:/datalog"
            ],
            'networks': ['kafka-net']
        }
    return zk_svcs

def make_kafka_services(n):
    kafka_svcs = {}
    for i in range(1, n+1):
        kafka_svcs[f'kafka{i}'] = {
            'image': 'bitnami/kafka:3.6',
            'container_name': f'kafka{i}',
            'hostname': f'kafka{i}',
            'ports': [f"{9091+i}:{9092+i-1}"],
            'environment': {
                'KAFKA_BROKER_ID': i,
                'KAFKA_CFG_ZOOKEEPER_CONNECT': ",".join([f"zookeeper{j}:2181" for j in range(1, 4)]),
                'KAFKA_LISTENERS': f"INTERNAL://kafka{i}:{29092+i-1},EXTERNAL://0.0.0.0:{9091+i}",
                'KAFKA_ADVERTISED_LISTENERS': f"INTERNAL://kafka{i}:{29092+i-1},EXTERNAL://${{KAFKA_PUBLIC_IP}}:{9091+i}",
                'KAFKA_LISTENER_SECURITY_PROTOCOL_MAP': 'INTERNAL:PLAINTEXT,EXTERNAL:PLAINTEXT',
                'KAFKA_CFG_INTER_BROKER_LISTENER_NAME': 'INTERNAL',
                'KAFKA_CFG_DEFAULT_REPLICATION_FACTOR': min(3, n),
                'KAFKA_CFG_MIN_INSYNC_REPLICAS': max(1, min(3, n)-1),
                'KAFKA_CFG_AUTO_CREATE_TOPICS_ENABLE': "false",
                'KAFKA_CFG_MESSAGE_MAX_BYTES': 5242880,
                'KAFKA_CFG_REPLICA_FETCH_MAX_BYTES': 5242880,
                'KAFKA_CFG_LOG_DIRS': '/bitnami/kafka/data/'
            },
            'volumes': [f"kafka{i}_data:/bitnami/kafka/data"],
            'networks': ['kafka-net'],
            'depends_on': [f'zookeeper{j}' for j in range(1, 4)]
        }
    return kafka_svcs

def main():
    parser = argparse.ArgumentParser(description="Generate docker-compose.yml for Kafka cluster")
    parser.add_argument('-b', '--kafka', type=int, default=3,
                        help="Số lượng Kafka brokers (mặc định 3)")
    parser.add_argument('-o', '--output', default='docker-compose.yaml',
                        help="Tên file output (mặc định docker-compose.yml)")
    args = parser.parse_args()

    compose = {
        'version': '3.8',
        'services': {},
        'networks': {'kafka-net': {'driver': 'bridge'}},
        'volumes': {}
    }

    # Thêm 3 ZooKeeper
    compose['services'].update(make_zk_services())
    for i in range(1, 4):
        compose['volumes'][f'zookeeper{i}_data'] = None
        compose['volumes'][f'zookeeper{i}_log'] = None

    # Thêm N Kafka brokers
    compose['services'].update(make_kafka_services(args.kafka))
    for i in range(1, args.kafka+1):
        compose['volumes'][f'kafka{i}_data'] = None

    # (Tuỳ chọn) thêm Kafka UI
    compose['services']['kafka-ui'] = {
        'image': 'provectuslabs/kafka-ui:latest',
        'container_name': 'kafka-ui',
        'ports': ['8080:8080'],
        'environment': {
            'KAFKA_CLUSTERS_0_NAME': 'ReID Kafka Cluster',
            'KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS': ",".join(
                [f"kafka{j}:{29092+j-1}" for j in range(1, args.kafka+1)]
            ),
            'KAFKA_CLUSTERS_0_ZOOKEEPER': ",".join(
                [f"zookeeper{j}:2181" for j in range(1, 4)]
            )
        },
        'networks': ['kafka-net'],
        'depends_on': [f'kafka{j}' for j in range(1, args.kafka+1)]
    }

    with open(args.output, 'w') as f:
        yaml.dump(compose, f, sort_keys=False)

    print(f"Generated {args.output} with 3 ZooKeeper and {args.kafka} Kafka brokers.")

if __name__ == '__main__':
    main()
