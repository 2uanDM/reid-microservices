#!/bin/bash

# Default number of consumers
NUM_CONSUMERS=${1:-3}

echo "Starting $NUM_CONSUMERS ReID consumers..."

# Array to store background process PIDs
pids=()

# Function to cleanup background processes on script exit
cleanup() {
    echo "Stopping all consumers..."
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping consumer with PID: $pid"
            kill -TERM "$pid"
        fi
    done
    wait
    echo "All consumers stopped."
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

# Start consumers in background
for i in $(seq 1 $NUM_CONSUMERS); do
    echo "Starting consumer $i..."
    python -m src &
    pid=$!
    pids+=($pid)
    echo "Consumer $i started with PID: $pid"
    
    # Small delay to avoid startup conflicts
    sleep 1
done

echo "All $NUM_CONSUMERS consumers started successfully!"
echo "PIDs: ${pids[*]}"
echo "Press Ctrl+C to stop all consumers"

# Wait for all background processes
wait
