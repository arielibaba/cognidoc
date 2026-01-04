"""
Object detection and extraction from images using YOLO.

This module provides:
- DetectionProcessor class for YOLO-based object detection
- Grouping of related detections (tables with captions, etc.)
- Fallback mechanism when YOLO detection fails

Fallback behavior:
- If YOLO fails to load or detect, the entire image is saved as text
- This ensures the pipeline continues even with detection issues
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from .utils.logger import logger, timer


def vertical_overlap(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> bool:
    """
    Check if two boxes have vertical overlap.

    Args:
        box1: First bounding box (x1, y1, x2, y2)
        box2: Second bounding box (x1, y1, x2, y2)

    Returns:
        True if boxes overlap vertically
    """
    overlap = min(box1[3], box2[3]) - max(box1[1], box2[1])
    return overlap > 0


class DetectionProcessor:
    """
    Process images with YOLO detection and grouping.

    Detects tables, pictures, text, and captions, then groups related elements.
    Includes fallback mechanism when detection fails.
    """

    TABLE_PICTURE_LABELS = {"table", "picture"}

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.2,
        iou_threshold: float = 0.8,
        high_quality: bool = True,
        enable_fallback: bool = True
    ):
        """
        Initialize the detection processor.

        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            high_quality: Use high quality JPEG compression
            enable_fallback: Enable fallback when detection fails
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.high_quality = high_quality
        self.enable_fallback = enable_fallback
        self.model = None
        self._model_loaded = False
        self._load_model()

    def _load_model(self):
        """Load the YOLO model with error handling."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self._model_loaded = True
            logger.info(f"YOLO model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self._model_loaded = False
            if not self.enable_fallback:
                raise

    def _fallback_process(self, image: np.ndarray, image_path: Path) -> Tuple[np.ndarray, List, List]:
        """
        Fallback processing when YOLO fails.

        Treats the entire image as a text region.

        Args:
            image: Image array
            image_path: Path to the image

        Returns:
            Tuple of (image, empty groups, single detection for entire image)
        """
        logger.warning(f"Using fallback for {image_path.name}: treating entire image as text")

        h, w = image.shape[:2]
        # Create a single detection covering the entire image as text
        fallback_detection = {
            "box": (0, 0, w, h),
            "label": "text",
            "index": 0
        }

        return image, [], [fallback_detection]

    def process_image(
        self,
        image_path: Path
    ) -> Tuple[np.ndarray, List[Dict], List[Dict]]:
        """
        Run inference on the given image and perform detection grouping.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (image array, grouped table/picture detections, other detections)
        """
        with timer(f"process {image_path.name}"):
            image = cv2.imread(str(image_path))

            if image is None:
                logger.error(f"Failed to read image: {image_path}")
                raise ValueError(f"Cannot read image: {image_path}")

            # Use fallback if model not loaded
            if not self._model_loaded:
                return self._fallback_process(image, image_path)

            try:
                # Run inference
                results = self.model(
                    str(image_path),
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
                result = results[0]

                # Check if any detections were made
                if len(result.boxes) == 0:
                    logger.warning(f"No detections in {image_path.name}, using fallback")
                    if self.enable_fallback:
                        return self._fallback_process(image, image_path)
                    return image, [], []

            except Exception as e:
                logger.error(f"YOLO inference failed for {image_path.name}: {e}")
                if self.enable_fallback:
                    return self._fallback_process(image, image_path)
                raise

            # Collect detections
            detections = []
            for i, (box, cls_idx) in enumerate(zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist())):
                x1, y1, x2, y2 = map(int, box)
                label = result.names[int(cls_idx)] if result.names else f"class_{int(cls_idx)}"
                detections.append({
                    "box": (x1, y1, x2, y2),
                    "label": label,
                    "index": i
                })

            # Sort detections by vertical (y1) then horizontal (x1) position
            sorted_detections = sorted(detections, key=lambda d: (d["box"][1], d["box"][0]))

            # Group detections for Table/Picture items
            grouped_indices = set()
            table_picture_groups = []
            others = []

            for i, det in enumerate(sorted_detections):
                if i in grouped_indices:
                    continue

                label_lower = det["label"].lower()

                if label_lower in self.TABLE_PICTURE_LABELS:
                    group_items = [det]
                    grouped_indices.add(i)

                    # Look for groupable items (Caption/Text)
                    for j, candidate in enumerate(sorted_detections):
                        if j == i or j in grouped_indices:
                            continue

                        candidate_label = candidate["label"].lower()

                        # Caption: group if adjacent or overlapping
                        if candidate_label == "caption":
                            if j == i - 1 or j == i + 1 or vertical_overlap(det["box"], candidate["box"]):
                                group_items.append(candidate)
                                grouped_indices.add(j)

                        # Text: group if before or overlapping
                        elif candidate_label == "text":
                            if j == i - 1 or vertical_overlap(det["box"], candidate["box"]):
                                group_items.append(candidate)
                                grouped_indices.add(j)

                    # Compute union bounding box
                    group_x1 = min(item["box"][0] for item in group_items)
                    group_y1 = min(item["box"][1] for item in group_items)
                    group_x2 = max(item["box"][2] for item in group_items)
                    group_y2 = max(item["box"][3] for item in group_items)

                    table_picture_groups.append({
                        "group_box": (group_x1, group_y1, group_x2, group_y2),
                        "label": det["label"],
                        "index": det["index"]
                    })
                else:
                    if i not in grouped_indices:
                        others.append(det)

            logger.debug(
                f"{image_path.name}: {len(table_picture_groups)} groups, "
                f"{len(others)} other detections"
            )

            return image, table_picture_groups, others

    def save_detections(
        self,
        image: np.ndarray,
        image_path: Path,
        table_picture_groups: List[Dict],
        others: List[Dict],
        output_dir: Path
    ) -> Dict[str, int]:
        """
        Save detected regions as separate images.

        Args:
            image: Source image array
            image_path: Path to source image
            table_picture_groups: Grouped table/picture detections
            others: Other detections (text, etc.)
            output_dir: Output directory

        Returns:
            Statistics dictionary
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        quality = 95 if self.high_quality else 75
        stats = {"tables": 0, "pictures": 0, "text": 0}

        # Save each Table/Picture group
        for count, group in enumerate(table_picture_groups, start=1):
            x1, y1, x2, y2 = group["group_box"]

            # Validate bounds
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Invalid crop bounds for {group['label']}, skipping")
                continue

            crop = image[y1:y2, x1:x2]
            label = group['label'].capitalize()
            out_filename = f"{image_path.stem}_{label}_{count}.jpg"

            try:
                cv2.imwrite(
                    str(output_dir / out_filename),
                    crop,
                    [cv2.IMWRITE_JPEG_QUALITY, quality]
                )
                if "table" in label.lower():
                    stats["tables"] += 1
                else:
                    stats["pictures"] += 1
            except Exception as e:
                logger.error(f"Failed to save {out_filename}: {e}")

        # Composite remaining detections into "Text" image
        if others:
            h, w = image.shape[:2]
            composite = np.ones((h, w, 3), dtype=np.uint8) * 255

            for det in others:
                x1, y1, x2, y2 = det["box"]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 > x1 and y2 > y1:
                    crop = image[y1:y2, x1:x2]
                    composite[y1:y2, x1:x2] = crop

            text_filename = f"{image_path.stem}_Text.jpg"
            try:
                cv2.imwrite(
                    str(output_dir / text_filename),
                    composite,
                    [cv2.IMWRITE_JPEG_QUALITY, quality]
                )
                stats["text"] += 1
            except Exception as e:
                logger.error(f"Failed to save {text_filename}: {e}")

        return stats


def extract_objects_from_image(
    input_dir: str,
    output_dir: str,
    model_path: str,
    conf_threshold: float = 0.2,
    iou_threshold: float = 0.8,
    high_quality: bool = True,
    enable_fallback: bool = True
) -> Dict[str, int]:
    """
    Process all images in a directory with YOLO detection.

    Args:
        input_dir: Directory containing input images
        output_dir: Directory for output crops
        model_path: Path to YOLO model
        conf_threshold: Confidence threshold
        iou_threshold: IOU threshold
        high_quality: Use high quality compression
        enable_fallback: Enable fallback when detection fails

    Returns:
        Statistics dictionary with counts
    """
    processor = DetectionProcessor(
        model_path,
        conf_threshold,
        iou_threshold,
        high_quality=high_quality,
        enable_fallback=enable_fallback
    )

    input_path = Path(input_dir)
    total_stats = {"images": 0, "tables": 0, "pictures": 0, "text": 0, "errors": 0, "fallbacks": 0}

    logger.info(f"Processing images in {input_dir}...")

    for image_path in input_path.glob("*"):
        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        logger.info(f"Processing {image_path.name}...")
        total_stats["images"] += 1

        try:
            image, groups, others = processor.process_image(image_path)

            # Check if fallback was used
            if not processor._model_loaded or (len(groups) == 0 and len(others) == 1):
                total_stats["fallbacks"] += 1

            stats = processor.save_detections(image, image_path, groups, others, output_dir)

            total_stats["tables"] += stats.get("tables", 0)
            total_stats["pictures"] += stats.get("pictures", 0)
            total_stats["text"] += stats.get("text", 0)

        except Exception as e:
            logger.error(f"Failed to process {image_path.name}: {e}")
            total_stats["errors"] += 1

    logger.info(f"""
    Object Extraction Complete:
    - Images processed: {total_stats['images']}
    - Tables extracted: {total_stats['tables']}
    - Pictures extracted: {total_stats['pictures']}
    - Text regions: {total_stats['text']}
    - Fallbacks used: {total_stats['fallbacks']}
    - Errors: {total_stats['errors']}
    - Output: {output_dir}
    """)

    return total_stats
