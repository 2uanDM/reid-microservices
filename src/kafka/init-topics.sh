#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}[INIT] Starting Kafka topics initialization...${NC}"

# Wait for Kafka brokers to be ready
echo -e "${YELLOW}[INIT] Waiting for Kafka brokers to be ready...${NC}"

# Function to check if Kafka is ready
check_kafka_ready() {
    kafka-topics.sh --bootstrap-server kafka1:29092 --list > /dev/null 2>&1
    return $?
}

# Wait up to 120 seconds for Kafka to be ready
COUNTER=0
MAX_ATTEMPTS=60
while ! check_kafka_ready; do
    if [ $COUNTER -eq $MAX_ATTEMPTS ]; then
        echo -e "${RED}[ERROR] Kafka brokers not ready after 120 seconds. Exiting...${NC}"
        exit 1
    fi
    echo -e "${YELLOW}[INIT] Waiting for Kafka... (attempt $((COUNTER+1))/$MAX_ATTEMPTS)${NC}"
    sleep 2
    COUNTER=$((COUNTER+1))
done

echo -e "${GREEN}[INIT] Kafka brokers are ready!${NC}"

# Define topics for ReID microservices
declare -A TOPICS=(
    # ReID input topic
    ["reid_input"]="6"           # ReID input
    # ReID output topic
    ["reid_output"]="6"          # ReID output
)

# Create topics
echo -e "${YELLOW}[INIT] Creating topics...${NC}"

for TOPIC in "${!TOPICS[@]}"; do
    PARTITIONS=${TOPICS[$TOPIC]}
    
    # Check if topic already exists
    if kafka-topics.sh --bootstrap-server kafka1:29092 --list | grep -q "^${TOPIC}$"; then
        echo -e "${YELLOW}[INIT] Topic '${TOPIC}' already exists, skipping...${NC}"
    else
        echo -e "${YELLOW}[INIT] Creating topic '${TOPIC}' with ${PARTITIONS} partitions...${NC}"
        
        if kafka-topics.sh --bootstrap-server kafka1:29092 \
            --create \
            --topic "${TOPIC}" \
            --partitions "${PARTITIONS}" \
            --replication-factor 3 \
            --config min.insync.replicas=2 \
            --config cleanup.policy=delete \
            --config retention.ms=604800000; then # 7 days retention
            echo -e "${GREEN}[INIT] Successfully created topic '${TOPIC}'${NC}"
        else
            echo -e "${RED}[ERROR] Failed to create topic '${TOPIC}'${NC}"
        fi
    fi
done

echo -e "${GREEN}[INIT] Topic initialization completed!${NC}"

# List all topics
echo -e "${YELLOW}[INIT] Current topics:${NC}"
kafka-topics.sh --bootstrap-server kafka1:29092 --list

# Show topic details
echo -e "${YELLOW}[INIT] Topic details:${NC}"
for TOPIC in "${!TOPICS[@]}"; do
    echo -e "${YELLOW}[INIT] Details for topic '${TOPIC}':${NC}"
    kafka-topics.sh --bootstrap-server kafka1:29092 --describe --topic "${TOPIC}"
    echo ""
done

echo -e "${GREEN}[INIT] Kafka topics initialization script completed successfully!${NC}" 