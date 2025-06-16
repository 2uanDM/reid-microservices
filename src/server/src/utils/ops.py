from typing import List, Tuple, Union

import cv2
import numpy as np
import torch


def xywh2ltwh(x: list):
    """
    Convert bounding box format from [x, y, w, h] to [x1, y1, w, h] where x1, y1 are top-left coordinates.

    Args:
        x (list): Input bounding box coordinates in xywh format.

    Returns:
        (list): Bounding box coordinates in xyltwh format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = x[0] - x[2] / 2  # top left x
    y[1] = x[1] - x[3] / 2  # top left y
    return y


def xyxy2xywh(x: list):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (list): Input bounding box coordinates in (x1, y1, x2, y2) format.

    Returns:
        (list): Bounding box coordinates in (x, y, width, height) format.
    """
    assert len(x) == 4, (
        f"input shape last dimension expected 4 but input shape is {len(x)}"
    )
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = (x[0] + x[2]) / 2  # x center
    y[1] = (x[1] + x[3]) / 2  # y center
    y[2] = x[2] - x[0]  # width
    y[3] = x[3] - x[1]  # height
    return y


def xywh2xyxy(x: list):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.

    Args:
        x (list): Input bounding box coordinates in (x, y, width, height) format.

    Returns:
        (list): Bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert len(x) == 4, (
        f"input shape last dimension expected 4 but input shape is {len(x)}"
    )
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    xy = x[:2]  # centers
    wh = x[2:] / 2  # half width-height
    y[:2] = xy - wh  # top left xy
    y[2:] = xy + wh  # bottom right xy
    return y


def _get_covariance_matrix(
    boxes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate covariance matrix from oriented bounding boxes.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (torch.Tensor): Covariance matrices corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def batch_probiou(
    obb1: Union[torch.Tensor, np.ndarray],
    obb2: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Calculate the probabilistic IoU between oriented bounding boxes.

    Args:
        obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing obb similarities.

    References:
        https://arxiv.org/pdf/2106.06072v1.pdf
    """
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2))
        / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (
        ((c1 + c2) * (x2 - x1) * (y1 - y2))
        / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (
            4
            * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt()
            + eps
        )
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd


def bbox_ioa(
    box1: np.ndarray, box2: np.ndarray, iou: bool = False, eps: float = 1e-7
) -> np.ndarray:
    """
    Calculate the intersection over box2 area given box1 and box2.

    Args:
        box1 (np.ndarray): A numpy array of shape (N, 4) representing N bounding boxes in x1y1x2y2 format.
        box2 (np.ndarray): A numpy array of shape (M, 4) representing M bounding boxes in x1y1x2y2 format.
        iou (bool, optional): Calculate the standard IoU if True else return inter_area/box2_area.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (np.ndarray): A numpy array of shape (N, M) representing the intersection over box2 area.
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (
        np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)
    ).clip(0) * (
        np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
    ).clip(0)

    # Box2 area
    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if iou:
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area = area + box1_area[:, None] - inter_area

    # Intersection over box2 area
    return inter_area / (area + eps)


def crop_image(image: np.ndarray, bboxes: List[List[float]]) -> List[np.ndarray]:
    """
    Crop an image based on a list of bounding boxes.
    """
    cropped_images = []
    for bbox in bboxes:
        # Convert float coordinates to integers
        x1, y1, x2, y2 = map(int, bbox)
        cropped_images.append(image[y1:y2, x1:x2])
    return cropped_images


def draw_bbox(
    image: np.ndarray,
    bboxes: List[List[float]],
    color: Tuple[int, int, int] = (139, 69, 19),
    thickness: int = 2,
    detection_confs: List[float] = None,
    ids: List[int] = None,
    genders: List[str] = None,
    gender_confs: List[float] = None,
    font_scale: float = 1,
    font_thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding boxes on an image with optional labels.

    Args:
        image (np.ndarray): Input image in BGR format.
        bboxes (List[List[float]]): List of bounding boxes in [x1, y1, x2, y2] format.
        color (Tuple[int, int, int], optional): Color of the bounding box in BGR format. Defaults to (0, 255, 0) green.
        thickness (int, optional): Thickness of the bounding box lines. Defaults to 2.
        confidences (List[float], optional): List of confidence scores for each bounding box. If provided, will be displayed as labels.
        ids (List[int], optional): List of object IDs for each bounding box. If provided, will be displayed as labels.
        class_names (List[str], optional): List of class names corresponding to class_ids. If not provided, class_ids will be shown as numbers.
        font_scale (float, optional): Font scale for text labels. Defaults to 0.5.
        font_thickness (int, optional): Font thickness for text labels. Defaults to 1.

    Returns:
        np.ndarray: Image with drawn bounding boxes and optional labels.
    """
    # Assert if genders exist then gender_confs must exist
    if genders is not None:
        if gender_confs is None:
            raise ValueError("gender_confs must be provided if genders is provided")

    # Create a copy to avoid modifying the original image
    image_copy = image.copy()

    for i, bbox in enumerate(bboxes):
        # Convert float coordinates to integers and ensure they're within image bounds
        x1, y1, x2, y2 = map(int, bbox)

        # Ensure coordinates are within image bounds
        h, w = image_copy.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        # Draw rectangle
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, thickness)

        # Prepare label text
        label_parts = []

        # Add ID if provided
        if ids is not None and i < len(ids):
            label_parts.append(f"ID:{ids[i]}")

        if genders is not None and i < len(genders):
            label_parts.append(f"{genders[i]} {gender_confs[i]:.2f}")

        # Add confidence if provided
        if detection_confs is not None and i < len(detection_confs):
            label_parts.append(f"det: {detection_confs[i]:.2f}")

        # Draw label if any parts exist
        if label_parts:
            label_text = " | ".join(label_parts)

            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            # Calculate label position (above the bounding box)
            label_x = x1 - 10
            label_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10

            # Draw background rectangle for text
            cv2.rectangle(
                image_copy,
                (label_x, label_y - text_height - 5),
                (label_x + text_width + 5, label_y + 5),
                color,
                -1,  # Filled rectangle
            )

            # Draw text
            cv2.putText(
                image_copy,
                label_text,
                (label_x + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),  # White text
                font_thickness,
            )

    return image_copy
