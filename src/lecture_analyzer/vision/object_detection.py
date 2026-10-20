"""Object detection service specifically identifying cell phones and contextual objects."""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from lecture_analyzer.config import ModelsConfig, VisionConfig
from lecture_analyzer.utils.paths import resolve_path
from lecture_analyzer.utils.validation import compute_iou, validate_bbox

logger = logging.getLogger("lecture_analyzer.vision.object_detection")


@dataclass
class DetectedObject:
    """Represents a detected object such as a cell phone, laptop, or book."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)


class ObjectDetector:
    """Manages YOLOv8 object detection with emphasis on phone/device presence."""

    def __init__(self, models_cfg: Optional[ModelsConfig] = None, vision_cfg: Optional[VisionConfig] = None):
        self.models_cfg = models_cfg or ModelsConfig()
        self.vision_cfg = vision_cfg or VisionConfig()
        self.model = None
        self.is_initialized = False

        self._initialize_model()

    def _initialize_model(self) -> None:
        model_path = resolve_path(self.models_cfg.yolo_detect_path)
        if not model_path.exists():
            logger.warning(
                f"YOLO detection model not found at {model_path}. "
                "Object detection will run in fallback (disabled) mode."
            )
            self.is_initialized = False
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(str(model_path))
            self.is_initialized = True
            logger.info(f"YOLOv8-detect initialized successfully from {model_path.name}.")
        except Exception as e:
            logger.error(f"Failed to load YOLO detection model: {e}", exc_info=True)
            self.is_initialized = False

    def process_frame(self, frame_bgr: np.ndarray) -> List[DetectedObject]:
        """Detects objects in the frame, filtering for relevant classes."""
        if not self.is_initialized or self.model is None or frame_bgr is None:
            return []

        h, w = frame_bgr.shape[:2]
        if h == 0 or w == 0:
            return []

        device = self.models_cfg.device
        if device == "auto":
            try:
                import torch
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        # COCO IDs: 67 = cell phone, 63 = laptop, 73 = book
        target_classes = [self.vision_cfg.phone_class_id, 63, 73]

        try:
            results = self.model.predict(
                source=frame_bgr,
                conf=self.vision_cfg.object_confidence_threshold,
                classes=target_classes,
                verbose=False,
                device=device,
            )
        except Exception as e:
            logger.debug(f"Object detection inference error: {e}")
            return []

        if not results or len(results) == 0:
            return []

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []

        detected: List[DetectedObject] = []
        boxes_xyxy = res.boxes.xyxy.cpu().numpy()
        boxes_conf = res.boxes.conf.cpu().numpy()
        boxes_cls = res.boxes.cls.cpu().numpy()
        names = res.names

        for i in range(len(boxes_xyxy)):
            cls_id = int(boxes_cls[i])
            conf = float(boxes_conf[i])
            raw_bbox = tuple(boxes_xyxy[i])
            valid_bbox = validate_bbox(raw_bbox, w, h)
            if valid_bbox is None:
                continue

            cls_name = names.get(cls_id, f"class_{cls_id}")
            detected.append(
                DetectedObject(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=round(conf, 3),
                    bbox=valid_bbox,
                )
            )

        return detected

    def is_phone_associated_with_person(
        self,
        phone_bbox: Tuple[int, int, int, int],
        person_bbox: Tuple[int, int, int, int]
    ) -> bool:
        """Determines if a detected phone is spatially associated with a person.

        A phone is considered associated if it overlaps with the person's bounding box
        or is located within the immediate upper/mid body region (hands/desk vicinity).
        """
        px1, py1, px2, py2 = person_bbox
        ox1, oy1, ox2, oy2 = phone_bbox

        # Expand person bbox slightly at the bottom (desk area)
        expanded_person = (
            px1 - 20,
            py1,
            px2 + 20,
            py2 + 50
        )

        iou = compute_iou(phone_bbox, expanded_person)
        if iou > 0.05:
            return True

        # Check containment of phone center inside person bbox
        phone_cx = (ox1 + ox2) / 2.0
        phone_cy = (oy1 + oy2) / 2.0
        if px1 <= phone_cx <= px2 and py1 <= phone_cy <= py2 + 50:
            return True

        return False
