"""
gesture_utils.py
Hand gesture recognition using MediaPipe Hands landmark detection, then a
simple rule-based classifier over which fingers are extended. This needs
no extra trained model beyond MediaPipe's bundled hand landmarker, which
keeps setup to a single `pip install mediapipe`.

Recognized gestures: Open Palm, Fist, Thumbs Up, Pointing, Peace / V Sign,
Call Me. Anything else is reported as "Unknown Gesture" (still detected —
just not classified — so the dashboard/live feed can show a bounding box).

If `mediapipe` isn't installed, GestureDetector quietly disables itself
(is_ready == False) and detect() returns [] instead of raising, so the
live feed keeps running with recognition/motion/mood detection even if
gesture detection hasn't been set up.

IMPORTANT — version pin: this uses MediaPipe's legacy `mp.solutions.hands`
Python API. Newer mediapipe releases (0.10.2x+) removed that module in
favor of the newer Tasks API, so install exactly the version pinned in
requirements.txt (`mediapipe==0.10.14`) — `pip install --upgrade mediapipe`
will break this file. If a future rewrite to the Tasks API is wanted, the
detection call becomes `HandLandmarker.detect()` instead of `Hands.process()`.
"""

import cv2

import config

try:
    import mediapipe as mp
    _MEDIAPIPE_AVAILABLE = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
except Exception:
    _MEDIAPIPE_AVAILABLE = False


_FINGER_TIPS = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky
_FINGER_PIPS = [3, 6, 10, 14, 18]
_HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS if _MEDIAPIPE_AVAILABLE else None


class GestureDetector:
    def __init__(self):
        self.enabled = _MEDIAPIPE_AVAILABLE
        self._hands = None
        if self.enabled:
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=config.GESTURE_MAX_HANDS,
                min_detection_confidence=config.GESTURE_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.GESTURE_TRACKING_CONFIDENCE,
            )

    @property
    def is_ready(self):
        return self.enabled

    def detect(self, bgr_frame):
        """
        Returns a list of dicts, one per detected hand:
            {"gesture": str, "landmarks": [(x, y), ...], "bbox": (x, y, w, h)}
        Returns [] if MediaPipe isn't installed or no hands are found.
        """
        if not self.enabled:
            return []

        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)

        detections = []
        if not results.multi_hand_landmarks:
            return detections

        handedness_list = results.multi_handedness or []
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = "Right"
            if i < len(handedness_list):
                label = handedness_list[i].classification[0].label

            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
            gesture = self._classify(points, label)

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            bbox = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

            detections.append({"gesture": gesture, "landmarks": points, "bbox": bbox})

        return detections

    def draw_landmarks(self, frame, landmarks):
        """Optional helper: draws the 21-point hand skeleton on `frame`."""
        if _HAND_CONNECTIONS is None:
            return
        for (x1, y1) in landmarks:
            cv2.circle(frame, (x1, y1), 3, (0, 217, 192), -1)
        for a, b in _HAND_CONNECTIONS:
            cv2.line(frame, landmarks[a], landmarks[b], (0, 217, 192), 1)

    def _classify(self, points, handedness_label):
        """Rule-based gesture classification from finger-extension state."""
        thumb, index, middle, ring, pinky = self._fingers_up(points, handedness_label)
        total_up = sum([thumb, index, middle, ring, pinky])

        if total_up == 0:
            return "Fist"
        if total_up == 5:
            return "Open Palm"
        if thumb and not any([index, middle, ring, pinky]):
            return "Thumbs Up"
        if index and middle and not any([thumb, ring, pinky]):
            return "Peace / V Sign"
        if index and not any([thumb, middle, ring, pinky]):
            return "Pointing"
        if thumb and pinky and not any([index, middle, ring]):
            return "Call Me"
        return "Unknown Gesture"

    def _fingers_up(self, points, handedness_label):
        """Returns [thumb, index, middle, ring, pinky] as 0/1 (1 = extended)."""
        fingers = []

        # Thumb: compare tip.x to pip.x; direction flips with handedness
        # because MediaPipe's "Right"/"Left" label is mirror-corrected for
        # a selfie-view camera (which is how the live feed is displayed).
        if handedness_label == "Right":
            fingers.append(int(points[_FINGER_TIPS[0]][0] > points[_FINGER_PIPS[0]][0]))
        else:
            fingers.append(int(points[_FINGER_TIPS[0]][0] < points[_FINGER_PIPS[0]][0]))

        # Other four fingers: tip above pip (smaller y) = extended
        for tip, pip in zip(_FINGER_TIPS[1:], _FINGER_PIPS[1:]):
            fingers.append(int(points[tip][1] < points[pip][1]))

        return fingers

    def close(self):
        if self._hands is not None:
            self._hands.close()