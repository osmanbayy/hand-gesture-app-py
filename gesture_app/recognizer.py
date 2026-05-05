import time
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .domain import GestureResult
from .model_loader import ensure_hand_landmarker_model
from .text_constants import (
    GESTURE_CALL_ME,
    GESTURE_FIST,
    GESTURE_FOUR,
    GESTURE_GUN,
    GESTURE_OK,
    GESTURE_OPEN_HAND,
    GESTURE_PINCH,
    GESTURE_POINT,
    GESTURE_ROCK,
    GESTURE_SWIPE_DOWN,
    GESTURE_SWIPE_LEFT,
    GESTURE_SWIPE_RIGHT,
    GESTURE_SWIPE_UP,
    GESTURE_THREE,
    GESTURE_THUMBS_DOWN,
    GESTURE_THUMBS_UP,
    GESTURE_TWO,
    GESTURE_UNKNOWN,
    GESTURE_V,
)


class HandGestureRecognizer:
    def __init__(self) -> None:
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17),
        ]
        model_path = ensure_hand_landmarker_model()
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.motion_points = []

    @staticmethod
    def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _finger_open(self, landmarks, tip_idx: int, pip_idx: int) -> bool:
        return landmarks[tip_idx].y < landmarks[pip_idx].y

    @staticmethod
    def _thumb_open(landmarks, scale: float) -> bool:
        thumb_tip = (landmarks[4].x, landmarks[4].y)
        index_mcp = (landmarks[5].x, landmarks[5].y)
        pinky_mcp = (landmarks[17].x, landmarks[17].y)
        thumb_to_index = float(np.hypot(thumb_tip[0] - index_mcp[0], thumb_tip[1] - index_mcp[1]))
        thumb_to_pinky = float(np.hypot(thumb_tip[0] - pinky_mcp[0], thumb_tip[1] - pinky_mcp[1]))
        return (thumb_to_index / scale > 0.55) or (thumb_to_pinky / scale > 1.1)

    def _detect_motion_gesture(
        self,
        wrist: Tuple[float, float],
        hand_scale: float,
        open_count: int,
    ) -> Optional[GestureResult]:
        now = time.time()
        self.motion_points.append((now, wrist[0], wrist[1]))
        self.motion_points = [p for p in self.motion_points if now - p[0] <= 0.45]
        if len(self.motion_points) < 4:
            return None

        first = self.motion_points[0]
        last = self.motion_points[-1]
        dx = last[1] - first[1]
        dy = last[2] - first[2]
        motion_threshold = max(0.10, 0.42 * hand_scale)

        if open_count >= 3:
            if abs(dx) > motion_threshold and abs(dx) > abs(dy) * 1.35:
                return GestureResult(GESTURE_SWIPE_RIGHT if dx > 0 else GESTURE_SWIPE_LEFT, 0.89)
            if abs(dy) > motion_threshold and abs(dy) > abs(dx) * 1.35:
                return GestureResult(GESTURE_SWIPE_DOWN if dy > 0 else GESTURE_SWIPE_UP, 0.87)

        return None

    def classify(self, landmarks) -> GestureResult:
        finger_open_states = [
            self._finger_open(landmarks, 8, 6),
            self._finger_open(landmarks, 12, 10),
            self._finger_open(landmarks, 16, 14),
            self._finger_open(landmarks, 20, 18),
        ]

        wrist = (landmarks[0].x, landmarks[0].y)
        thumb_tip = (landmarks[4].x, landmarks[4].y)
        index_tip = (landmarks[8].x, landmarks[8].y)
        middle_tip = (landmarks[12].x, landmarks[12].y)
        ring_tip = (landmarks[16].x, landmarks[16].y)
        pinky_tip = (landmarks[20].x, landmarks[20].y)

        hand_scale = self._distance(
            (landmarks[5].x, landmarks[5].y),
            (landmarks[17].x, landmarks[17].y),
        ) + 1e-6

        thumb_is_open = self._thumb_open(landmarks, hand_scale)
        thumb_up_metric = (wrist[1] - thumb_tip[1]) / hand_scale
        thumb_down_metric = (thumb_tip[1] - wrist[1]) / hand_scale
        index_thumb_distance = self._distance(index_tip, thumb_tip) / hand_scale
        middle_thumb_distance = self._distance(middle_tip, thumb_tip) / hand_scale
        index_middle_distance = self._distance(index_tip, middle_tip) / hand_scale
        ring_thumb_distance = self._distance(ring_tip, thumb_tip) / hand_scale
        pinky_thumb_distance = self._distance(pinky_tip, thumb_tip) / hand_scale

        open_count = sum(finger_open_states)
        index_open, middle_open, ring_open, pinky_open = finger_open_states

        motion_gesture = self._detect_motion_gesture(wrist, hand_scale, open_count)
        if motion_gesture is not None:
            return motion_gesture

        if index_thumb_distance < 0.26 and middle_open and ring_open and pinky_open:
            return GestureResult(GESTURE_OK, 0.93)
        if index_thumb_distance < 0.25 and not middle_open:
            return GestureResult(GESTURE_PINCH, 0.90)
        if (
            thumb_up_metric > 0.78
            and not index_open
            and not middle_open
            and not ring_open
            and not pinky_open
        ):
            return GestureResult(GESTURE_THUMBS_UP, 0.91)
        if (
            thumb_down_metric > 0.85
            and not index_open
            and not middle_open
            and not ring_open
            and not pinky_open
        ):
            return GestureResult(GESTURE_THUMBS_DOWN, 0.89)
        if index_open and middle_open and not ring_open and not pinky_open:
            if index_middle_distance > 0.34:
                return GestureResult(GESTURE_V, 0.90)
            return GestureResult(GESTURE_TWO, 0.84)
        if index_open and middle_open and ring_open and not pinky_open:
            return GestureResult(GESTURE_THREE, 0.88)
        if index_open and middle_open and ring_open and pinky_open and not thumb_is_open:
            return GestureResult(GESTURE_FOUR, 0.87)
        if (
            index_open
            and not middle_open
            and not ring_open
            and not pinky_open
            and thumb_is_open
        ):
            return GestureResult(GESTURE_GUN, 0.86)
        if (
            not index_open
            and not middle_open
            and not ring_open
            and pinky_open
            and thumb_is_open
        ):
            return GestureResult(GESTURE_CALL_ME, 0.88)
        if index_open and not middle_open and not ring_open and pinky_open:
            return GestureResult(GESTURE_ROCK, 0.87)
        if open_count == 4 and thumb_is_open and index_thumb_distance > 0.52:
            return GestureResult(GESTURE_OPEN_HAND, 0.93)
        if open_count == 0 and index_thumb_distance < 0.45 and middle_thumb_distance < 0.50:
            return GestureResult(GESTURE_FIST, 0.90)
        if (
            index_open
            and not middle_open
            and not ring_open
            and not pinky_open
            and not thumb_is_open
            and ring_thumb_distance > 0.65
            and pinky_thumb_distance > 0.65
        ):
            return GestureResult(GESTURE_POINT, 0.86)

        return GestureResult(GESTURE_UNKNOWN, 0.52)

    def process(self, frame_bgr: np.ndarray):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self.landmarker.detect(mp_image)
        return results.hand_landmarks

    def draw_hand(self, frame_bgr: np.ndarray, hand_landmarks) -> None:
        height, width, _ = frame_bgr.shape
        points = []
        for landmark in hand_landmarks:
            px = int(landmark.x * width)
            py = int(landmark.y * height)
            points.append((px, py))
            cv2.circle(frame_bgr, (px, py), 3, (66, 133, 244), -1)

        for start_idx, end_idx in self.hand_connections:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(frame_bgr, points[start_idx], points[end_idx], (15, 157, 88), 2)

    def close(self) -> None:
        self.landmarker.close()
