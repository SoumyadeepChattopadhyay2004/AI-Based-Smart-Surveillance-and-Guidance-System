"""
face_utils.py
Face detection (Haar Cascade) + face recognition (OpenCV LBPH) utilities.

Face samples, the trained model and the label map all live in MongoDB Atlas
via storage.py (GridFS + a small metadata doc) — nothing is kept on local
disk permanently. Training/loading briefly writes to a temp file because
cv2.face.LBPHFaceRecognizer only has a file-path API, and that temp file is
deleted immediately after use.
"""

import os
import tempfile
import cv2
import numpy as np

import config
import storage

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_faces(gray_frame):
    """Returns list of (x, y, w, h) bounding boxes for faces in a grayscale frame."""
    return _face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.15, minNeighbors=5, minSize=(60, 60)
    )


def capture_face_samples(user_id, camera_index=None, num_samples=None, frame_provider=None):
    """
    Captures `num_samples` cropped, grayscale face images and uploads each
    one straight to GridFS (storage.save_face_sample) — no local file is
    ever written. If `frame_provider` (a callable returning a BGR frame) is
    supplied it is used instead of opening a new camera — handy for reusing
    an already-open cv2.VideoCapture from a Flask video stream.
    """
    num_samples = num_samples or config.FACE_SAMPLES_PER_USER

    owns_camera = frame_provider is None
    cap = None
    if owns_camera:
        cap = cv2.VideoCapture(camera_index if camera_index is not None else config.CAMERA_INDEX)

    count = storage.count_face_samples(user_id)
    try:
        while count < num_samples:
            frame = frame_provider() if frame_provider else cap.read()[1]
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray)
            for (x, y, w, h) in faces:
                face_img = gray[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, (200, 200))
                ok, buffer = cv2.imencode(".jpg", face_img)
                if ok:
                    storage.save_face_sample(user_id, count, buffer.tobytes())
                    count += 1
                break  # one face per frame is enough
    finally:
        if owns_camera and cap is not None:
            cap.release()

    return count


def train_recognizer():
    """
    Trains an LBPH face recognizer on every sample stored in GridFS
    (grouped by user_id) and persists the trained model + label map back to
    MongoDB Atlas. Returns the number of users trained on.
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces, labels = [], []
    label_map = {}  # numeric label (int) -> user_id (int, matches DB users.id)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for user_id in storage.user_ids_with_samples():
            numeric_label = len(label_map)
            label_map[numeric_label] = user_id

            for path in storage.download_user_samples(user_id, tmp_dir):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                faces.append(img)
                labels.append(numeric_label)
        # every temp jpg is deleted automatically when this block exits

    if not faces:
        return 0

    recognizer.train(faces, np.array(labels))

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "trainer.yml")
        recognizer.save(model_path)
        with open(model_path, "rb") as f:
            storage.save_trainer_model(f.read())
        # tmp_dir + trainer.yml deleted automatically here

    storage.save_labels(label_map)
    return len(label_map)


class FaceRecognizer:
    """Thin wrapper that loads the trained model once and exposes .predict()."""

    def __init__(self):
        self.recognizer = None
        self.label_map = {}
        self.reload()

    def reload(self):
        self.label_map = storage.load_labels()
        model_bytes = storage.load_trainer_model_bytes()

        if model_bytes and self.label_map:
            with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as tmp:
                tmp.write(model_bytes)
                tmp_path = tmp.name
            try:
                self.recognizer = cv2.face.LBPHFaceRecognizer_create()
                self.recognizer.read(tmp_path)
            finally:
                os.remove(tmp_path)  # local copy only existed for this call
        else:
            self.recognizer = None

    @property
    def is_ready(self):
        return self.recognizer is not None

    def predict(self, gray_face_img):
        """
        Returns (user_id_or_None, confidence). Lower confidence = more sure.
        user_id is None when the face doesn't match any known label within
        the configured threshold (i.e. "Unknown person").
        """
        if not self.is_ready:
            return None, 999.0

        face_img = cv2.resize(gray_face_img, (200, 200))
        numeric_label, confidence = self.recognizer.predict(face_img)

        if confidence <= config.LBPH_CONFIDENCE_THRESHOLD:
            return self.label_map.get(numeric_label), confidence
        return None, confidence
