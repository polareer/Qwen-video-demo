"""Frame-level ergonomic signals from first-person video."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .schemas import Detection


@dataclass
class FrameAnalysis:
    frame_index: int
    timestamp_sec: float
    target: Detection
    hands: list[Detection] = field(default_factory=list)
    detections: list[Detection] = field(default_factory=list)
    brightness: float = 0.0
    sharpness: float = 0.0
    center_offset: float = 0.0
    occlusion_ratio: float = 0.0
    visibility_score: float = 1.0
    reachability_score: float = 1.0


class FrameAnalyzer:
    """Combines optional detectors with deterministic OpenCV heuristics."""

    def __init__(
        self,
        yolo_model_path: str | None = None,
        target_label: str | None = None,
        use_mediapipe_hands: bool = True,
    ) -> None:
        self.target_label = target_label
        self.yolo = self._load_yolo(yolo_model_path)
        self.hands_detector = self._load_mediapipe_hands(use_mediapipe_hands)

    def analyze(self, frame: np.ndarray, frame_index: int, timestamp_sec: float) -> FrameAnalysis:
        detections = self._detect_with_yolo(frame)
        hands = self._detect_hands(frame, detections)
        target = self._select_target(frame, detections)
        brightness = self._brightness(frame)
        sharpness = self._sharpness(frame)
        center_offset = self._center_offset(frame.shape, target.bbox)
        occlusion_ratio = self._occlusion_ratio(target.bbox, [hand.bbox for hand in hands])
        visibility_score = self._visibility_score(brightness, sharpness, center_offset, occlusion_ratio)
        reachability_score = self._reachability_score(frame.shape, target.bbox, [hand.bbox for hand in hands])
        return FrameAnalysis(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            target=target,
            hands=hands,
            detections=detections,
            brightness=brightness,
            sharpness=sharpness,
            center_offset=center_offset,
            occlusion_ratio=occlusion_ratio,
            visibility_score=visibility_score,
            reachability_score=reachability_score,
        )

    def _load_yolo(self, model_path: str | None) -> Any | None:
        if not model_path:
            return None
        try:
            from ultralytics import YOLO  # type: ignore

            return YOLO(model_path)
        except Exception:
            return None

    def _load_mediapipe_hands(self, enabled: bool) -> Any | None:
        if not enabled:
            return None
        try:
            import mediapipe as mp  # type: ignore

            return mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
            )
        except Exception:
            return None

    def _detect_with_yolo(self, frame: np.ndarray) -> list[Detection]:
        if self.yolo is None:
            return []
        results = self.yolo.predict(frame, verbose=False)
        detections: list[Detection] = []
        for result in results:
            names = getattr(result, "names", {})
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0].item())
                label = str(names.get(cls_id, cls_id))
                conf = float(box.conf[0].item())
                xyxy = [int(v) for v in box.xyxy[0].tolist()]
                detections.append(Detection(label=label, confidence=conf, bbox=xyxy, source="yolo"))
        return detections

    def _detect_hands(self, frame: np.ndarray, detections: list[Detection]) -> list[Detection]:
        yolo_hands = [det for det in detections if det.label.lower() in {"hand", "hands"}]
        if yolo_hands:
            return yolo_hands
        if self.hands_detector is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands_detector.process(rgb)
            if result.multi_hand_landmarks:
                h, w = frame.shape[:2]
                hands: list[Detection] = []
                for landmarks in result.multi_hand_landmarks:
                    xs = [int(point.x * w) for point in landmarks.landmark]
                    ys = [int(point.y * h) for point in landmarks.landmark]
                    hands.append(
                        Detection(
                            label="hand",
                            confidence=0.8,
                            bbox=[max(0, min(xs)), max(0, min(ys)), min(w - 1, max(xs)), min(h - 1, max(ys))],
                            source="mediapipe",
                        )
                    )
                return hands
        return self._detect_skin_hands(frame)

    def _select_target(self, frame: np.ndarray, detections: list[Detection]) -> Detection:
        if self.target_label:
            for det in detections:
                if det.label.lower() == self.target_label.lower():
                    return det
        non_hand = [det for det in detections if det.label.lower() not in {"person", "hand", "hands"}]
        if non_hand:
            return max(non_hand, key=lambda item: item.confidence)
        bbox = self._largest_edge_region(frame)
        return Detection(label=self.target_label or "operation_target", confidence=0.5, bbox=bbox)

    def _largest_edge_region(self, frame: np.ndarray) -> list[int]:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = max(200, int(w * h * 0.005))
        candidates = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            area = bw * bh
            if area >= min_area:
                candidates.append((area, [x, y, x + bw, y + bh]))
        if candidates:
            return max(candidates, key=lambda item: item[0])[1]
        margin_x = int(w * 0.32)
        margin_y = int(h * 0.30)
        return [margin_x, margin_y, w - margin_x, h - margin_y]

    def _detect_skin_hands(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 25, 40], dtype=np.uint8)
        upper = np.array([25, 180, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask[: int(h * 0.25), :] = 0
        mask = cv2.medianBlur(mask, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            area = bw * bh
            if area > w * h * 0.01:
                detections.append(Detection("hand", 0.35, [x, y, x + bw, y + bh], "skin_heuristic"))
        return detections[:2]

    def _brightness(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(gray.mean() / 255.0)

    def _sharpness(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        value = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(min(value / 500.0, 1.0))

    def _center_offset(self, shape: tuple[int, ...], bbox: list[int]) -> float:
        h, w = shape[:2]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        dx = abs(cx - w / 2) / (w / 2)
        dy = abs(cy - h / 2) / (h / 2)
        return float(min((dx * dx + dy * dy) ** 0.5, 1.0))

    def _occlusion_ratio(self, target_bbox: list[int], hand_boxes: list[list[int]]) -> float:
        target_area = max(1, (target_bbox[2] - target_bbox[0]) * (target_bbox[3] - target_bbox[1]))
        overlap = 0
        for box in hand_boxes:
            x1 = max(target_bbox[0], box[0])
            y1 = max(target_bbox[1], box[1])
            x2 = min(target_bbox[2], box[2])
            y2 = min(target_bbox[3], box[3])
            if x2 > x1 and y2 > y1:
                overlap += (x2 - x1) * (y2 - y1)
        return float(min(overlap / target_area, 1.0))

    def _visibility_score(self, brightness: float, sharpness: float, center_offset: float, occlusion_ratio: float) -> float:
        score = 1.0
        score -= max(0.0, 0.35 - brightness) * 1.2
        score -= max(0.0, 0.35 - sharpness) * 0.8
        score -= center_offset * 0.25
        score -= occlusion_ratio * 0.45
        return float(min(max(score, 0.0), 1.0))

    def _reachability_score(self, shape: tuple[int, ...], target_bbox: list[int], hand_boxes: list[list[int]]) -> float:
        if not hand_boxes:
            return 0.55
        h, w = shape[:2]
        target_cx = (target_bbox[0] + target_bbox[2]) / 2
        target_cy = (target_bbox[1] + target_bbox[3]) / 2
        best_distance = 1.0
        for box in hand_boxes:
            hand_cx = (box[0] + box[2]) / 2
            hand_cy = (box[1] + box[3]) / 2
            distance = (((hand_cx - target_cx) / w) ** 2 + ((hand_cy - target_cy) / h) ** 2) ** 0.5
            best_distance = min(best_distance, distance)
        edge_penalty = self._center_offset(shape, target_bbox) * 0.35
        score = 1.0 - min(best_distance * 1.8, 0.75) - edge_penalty
        return float(min(max(score, 0.0), 1.0))
