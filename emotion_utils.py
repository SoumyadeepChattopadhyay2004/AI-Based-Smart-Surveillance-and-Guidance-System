"""
emotion_utils.py
Facial expression / mood detection — Happy, Sad, Angry, Surprised, Anxious
(fearful), Disgusted, Neutral, Contempt — run through OpenCV's DNN module
on a small ONNX CNN (FER+, ~34 MB). This keeps the project dependency-light
and consistent with the rest of the app: no PyTorch/TensorFlow/dlib needed,
just OpenCV, which is already a hard requirement.

Setup (one-time):
    Download the model from the ONNX Model Zoo and save it at the path
    below (config.EMOTION_MODEL_PATH):
        https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx
    -> save as: models/emotion-ferplus-8.onnx

If the model file isn't present, EmotionDetector quietly disables itself
(is_ready == False) and detect_mood() returns ("Unknown", 0.0) instead of
raising, so the live feed keeps running with recognition/motion/gestures
even if mood detection hasn't been set up yet.
"""

import os
import cv2
import numpy as np

import config

# FER+ label order, as published by the model authors.
EMOTIONS = ["Neutral", "Happy", "Surprised", "Sad", "Angry", "Disgusted", "Anxious", "Contempt"]

# Colors (BGR) for overlaying each mood on the video frame.
EMOTION_COLORS = {
    "Neutral": (200, 200, 200),
    "Happy": (0, 217, 192),
    "Surprised": (0, 184, 255),
    "Sad": (255, 140, 0),
    "Angry": (68, 68, 255),
    "Disgusted": (0, 128, 128),
    "Anxious": (140, 0, 255),
    "Contempt": (128, 0, 128),
}


class EmotionDetector:
    """Thin wrapper around the FER+ ONNX model loaded via cv2.dnn."""

    def __init__(self, model_path=None):
        self.model_path = model_path or config.EMOTION_MODEL_PATH
        self.net = None
        if os.path.exists(self.model_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(self.model_path)
            except Exception:
                self.net = None

    @property
    def is_ready(self):
        return self.net is not None

    def detect_mood(self, gray_face_img):
        """
        gray_face_img: grayscale crop of a single face (any size — it's
        resized internally).
        Returns (label, confidence_0_to_1). Returns ("Unknown", 0.0) if the
        model isn't loaded or the crop is empty/invalid.
        """
        if self.net is None or gray_face_img is None or gray_face_img.size == 0:
            return "Unknown", 0.0

        try:
            face = cv2.resize(gray_face_img, (64, 64)).astype(np.float32)
        except cv2.error:
            return "Unknown", 0.0

        blob = face.reshape(1, 1, 64, 64)

        self.net.setInput(blob)
        scores = self.net.forward().flatten()

        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        idx = int(np.argmax(probs))
        return EMOTIONS[idx], float(probs[idx])