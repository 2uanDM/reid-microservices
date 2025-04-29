#!/bin/bash

source ../../.env

VIDEO_DIR="./videos"
IMAGE_NAME="edge-device"

#build docker image for each edge device
docker build -t $IMAGE_NAME .

VIDEO_FILES=($(ls $VIDEO_DIR/*.mp4))

#folder to log output
mkdir -p logs

run_container(){
    local idx=$1
    local video_path=${VIDEO_FILES[$idx]}
    local container_name="edge_cam_$idx"

    echo "Running container: $container_name for video: $video_path"

    #run container
    docker run --rm \
        --name $container_name \
        --cpus="1.0" \
        --memory="2g" \
        -e KAFKA_SERVER_URI=$KAFKA_SERVER_URI \
        -v "$(pwd)/$video_path":/app/input.mp4 \
        $IMAGE_NAME > "logs/$container_name.log" 2>&1 &
}

for i in "${!VIDEO_FILES[@]}"; do 
    run_container
done

wait

echo "Completed running"

#average fps for each container
echo -e "\n Average FPS of each edge device:"
for i in "${!VIDEO_FILES[@]}"; do
    log_file = "logs/edge_cam_$i.log"
    fps=$(grep -oP "FPS:\s*\K[0-9.]+" "$log_file" | tail -1)
    echo " Edge device $i (video: ${VIDEO_FILES[$i]}): ${fps:-not found} FPS"
done



