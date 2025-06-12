"""
This script is used to test the image quality, based on the values of cv2.IMWRITE_JPEG_QUALITY
"""

import os
import random
import shutil
import time

import cv2
import numpy as np

shutil.rmtree("frames", ignore_errors=True)
os.makedirs("frames", exist_ok=True)


def get_frame_size(frame: np.ndarray) -> int:
    """
    Get the size of the frame (in KB)
    """
    return frame.shape[0] * frame.shape[1] / 1024


def get_image_size(image_path: str) -> int:
    """
    Get the size of the image (in KB)
    """
    return os.path.getsize(image_path) / 1024


def test_image_quality():
    """
    Extract first 5 frames, each with different quality values
    """
    # Video path
    video_path = "/mnt/e/workspace/Dataset/thesis/turn2/edtech_bacony_low_1.mp4"

    # Open video
    cap = cv2.VideoCapture(video_path)

    # Skip first 3 seconds of the video
    cap.set(cv2.CAP_PROP_POS_MSEC, 26000)  # 3000 milliseconds = 3 seconds

    # Extract first 5 frames

    for i in range(10):
        print(f"Extracting frame {i}")
        ret, frame = cap.read()
        if not ret:
            break

        # Get frame size
        frame_size = get_frame_size(frame)

        # Save frame with different quality values
        for quality in range(10, 120, 10):
            # Save frame
            file_path = os.path.join(
                os.getcwd(), "frames", f"{i}", f"frame_{i}_quality_{quality}.jpg"
            )
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            cv2.imwrite(
                file_path,
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, quality],
            )

            # Get image size
            image_size = get_image_size(file_path)

            print(
                f"Frame {i} with quality {quality} saved to {file_path} - {frame_size} KB -> {image_size} KB"
            )

        time.sleep(random.randint(1, 3))

    # Close video
    cap.release()


if __name__ == "__main__":
    test_image_quality()
