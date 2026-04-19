"""Visualization utilities for drawing detection boxes on images."""

import numpy as np
import cv2

# A simple color palette for different classes
_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128),
    (128, 128, 0), (0, 128, 128), (128, 0, 128),
]


def _get_color(class_id):
    return _COLORS[class_id % len(_COLORS)]


def visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=20,
        min_score_thresh=0.5,
        line_thickness=4,
        **kwargs):
    """
    Draws bounding boxes and labels on an image array (in-place).

    Args:
        image: uint8 numpy array of shape (height, width, 3)
        boxes: float32 array of shape [N, 4] — [ymin, xmin, ymax, xmax]
        classes: int array of shape [N]
        scores: float32 array of shape [N]
        category_index: dict {id: {'id': id, 'name': name}}
        use_normalized_coordinates: if True, coords are in [0,1]
        max_boxes_to_draw: max number of boxes to draw
        min_score_thresh: only draw boxes with score >= this threshold
        line_thickness: thickness of bounding box lines
    """
    if image is None or len(image.shape) != 3:
        return

    height, width = image.shape[:2]
    num_boxes = min(len(boxes), max_boxes_to_draw)

    for i in range(num_boxes):
        score = scores[i] if scores is not None else 1.0
        if score < min_score_thresh:
            continue

        box = boxes[i]
        ymin, xmin, ymax, xmax = box

        if use_normalized_coordinates:
            left   = int(xmin * width)
            right  = int(xmax * width)
            top    = int(ymin * height)
            bottom = int(ymax * height)
        else:
            left, right, top, bottom = int(xmin), int(xmax), int(ymin), int(ymax)

        class_id = int(classes[i])
        color = _get_color(class_id)

        # Draw the bounding box
        cv2.rectangle(image, (left, top), (right, bottom), color, line_thickness)

        # Build label text
        label = category_index.get(class_id, {}).get('name', str(class_id))
        display_str = f"{label}: {int(100 * score)}%"

        # Draw label background
        (text_w, text_h), baseline = cv2.getTextSize(
            display_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image,
                      (left, top - text_h - baseline - 5),
                      (left + text_w, top),
                      color, -1)
        cv2.putText(image, display_str,
                    (left, top - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
