"""
motion_detector.py
Suspicious Movement Tracker — background-subtraction based motion detector.

Flags a frame as "suspicious" when the ratio of moving pixels to total
frame area exceeds a configurable threshold (fast / large movement is far
more likely to be someone running, a crowd, or a struggle than routine
walking past the camera).
"""

import time
import cv2
import numpy as np

import config


class MotionDetector:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=False
        )
        self._last_alert_time = 0.0

    def analyze(self, frame):
        """
        Runs background subtraction on `frame` and returns a dict:
            {
                "moving_boxes": [(x, y, w, h), ...],
                "area_ratio": float,        # fraction of frame covered by motion
                "is_suspicious": bool,
                "can_alert": bool           # True only if cooldown has elapsed
            }
        """
        fg_mask = self.bg_subtractor.apply(frame)
        fg_mask = cv2.medianBlur(fg_mask, 5)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_area = frame.shape[0] * frame.shape[1]
        moving_boxes = []
        total_motion_area = 0

        for c in contours:
            area = cv2.contourArea(c)
            if area < config.MOTION_MIN_CONTOUR_AREA:
                continue
            x, y, w, h = cv2.boundingRect(c)
            moving_boxes.append((x, y, w, h))
            total_motion_area += area

        area_ratio = total_motion_area / frame_area if frame_area else 0
        is_suspicious = area_ratio >= config.MOTION_SUSPICIOUS_AREA_RATIO

        can_alert = False
        now = time.time()
        if is_suspicious and (now - self._last_alert_time) >= config.MOTION_ALERT_COOLDOWN_SECONDS:
            can_alert = True
            self._last_alert_time = now

        return {
            "moving_boxes": moving_boxes,
            "area_ratio": round(area_ratio, 4),
            "is_suspicious": is_suspicious,
            "can_alert": can_alert,
        }