"""
config.py
Central configuration for the AI-Based Smart Surveillance and Guidance System.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Groq API configuration
# ---------------------------------------------------------------------------
# Get a free key from https://console.groq.com/keys
# You can either export it as an environment variable:
#     export GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxx"
# or simply paste it below (not recommended for shared / production code).
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "PASTE_YOUR_GROQ_API_KEY_HERE")

# Any current Groq-hosted model works; llama-3.3-70b-versatile is a strong,
# fast general purpose choice for turning raw event data into readable text.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# If no valid key is set, the system falls back to template-based alert text
# instead of calling the API, so the app still runs end-to-end out of the box.
GROQ_ENABLED = bool(GROQ_API_KEY) and GROQ_API_KEY != "PASTE_YOUR_GROQ_API_KEY_HERE"

# ---------------------------------------------------------------------------
# Flask configuration
# ---------------------------------------------------------------------------
SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-secret-key-in-production")
DEBUG = True
HOST = "0.0.0.0"
PORT = 5000

# ---------------------------------------------------------------------------
# Storage — MongoDB Atlas
# ---------------------------------------------------------------------------
# All persistent data (users, attendance, alerts, face samples, the trained
# LBPH model, alert snapshots) lives in Atlas — see mongo.py / storage.py /
# database.py. Nothing is written to local disk except brief temp files
# OpenCV's file-based model API requires, deleted immediately after use.
#
# Get a free cluster at https://www.mongodb.com/cloud/atlas/register, then
# either export it as an environment variable:
#     export MONGODB_URI="mongodb+srv://<user>:<password>@<cluster>.mongodb.net/"
# or paste it below (not recommended for shared / production code).
MONGODB_URI = os.environ.get(
    "MONGODB_URI", "mongodb+srv://soumyadeepchattopadhyay001_db_user:8VGFbHwRK2JadwXh@campus.2avrofm.mongodb.net/?appName=campus"
)
MONGODB_DB_NAME = os.environ.get("MONGODB_DB_NAME", "smart_surveillance")

# models/ ships the bundled ONNX emotion model as a static repo asset (not
# user-generated data), so it stays local.
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Camera / video settings
# ---------------------------------------------------------------------------
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", 0))
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Number of face samples captured per person during registration
FACE_SAMPLES_PER_USER = 30

# Recognition confidence threshold (LBPH: LOWER distance = more confident).
# Faces with a distance above this are treated as "Unknown".
LBPH_CONFIDENCE_THRESHOLD = 70

# ---------------------------------------------------------------------------
# Attendance settings
# ---------------------------------------------------------------------------
# Minimum minutes between two attendance marks for the same person on the
# same day, to avoid duplicate entries while someone lingers in front of camera.
ATTENDANCE_COOLDOWN_MINUTES = 1

# ---------------------------------------------------------------------------
# Unknown person alerting
# ---------------------------------------------------------------------------
# Consecutive frames an unrecognized face must appear in before an alert fires
UNKNOWN_FACE_ALERT_FRAME_THRESHOLD = 15
# Minimum seconds between two unknown-person alerts (avoid alert spam)
UNKNOWN_ALERT_COOLDOWN_SECONDS = 30

# ---------------------------------------------------------------------------
# Suspicious movement tracker settings
# ---------------------------------------------------------------------------
MOTION_MIN_CONTOUR_AREA = 3500       # ignore small/noisy movement
MOTION_SUSPICIOUS_AREA_RATIO = 0.18  # fraction of frame area that triggers "suspicious"
MOTION_ALERT_COOLDOWN_SECONDS = 20

# ---------------------------------------------------------------------------
# Mood / facial-expression detection (emotion_utils.py)
# ---------------------------------------------------------------------------
ENABLE_EMOTION_DETECTION = True
# Uses a small ONNX "FER+" CNN via cv2.dnn. Download once from the ONNX
# Model Zoo and place it at EMOTION_MODEL_PATH — see emotion_utils.py header
# for the exact URL. If the file is missing, mood detection quietly
# disables itself and the rest of the app is unaffected.
EMOTION_MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion-ferplus-8.onnx")

# Moods worth calling out with a dashboard alert when sustained on a face
# (e.g. a visibly distressed or angry visitor), plus streak/cooldown so a
# single flicker of a frame doesn't spam the alerts table.
MOOD_ALERT_EMOTIONS = {"Angry", "Anxious"}
MOOD_ALERT_FRAME_THRESHOLD = 15
MOOD_ALERT_COOLDOWN_SECONDS = 30
# Minimum seconds between routine mood_logs rows for the *same* person, so
# every single video frame doesn't get written to the DB.
MOOD_LOG_INTERVAL_SECONDS = 5

# ---------------------------------------------------------------------------
# Hand gesture detection (gesture_utils.py)
# ---------------------------------------------------------------------------
ENABLE_GESTURE_DETECTION = True
# Uses MediaPipe Hands (bundled model, no separate download needed).
GESTURE_MAX_HANDS = 2
GESTURE_DETECTION_CONFIDENCE = 0.6
GESTURE_TRACKING_CONFIDENCE = 0.5

# Gestures worth raising an alert for (e.g. a raised open palm held toward
# the camera as a "stop / help" signal), with the same streak/cooldown
# pattern used for unknown-person and motion alerts.
GESTURE_ALERT_GESTURES = {"Open Palm"}
GESTURE_ALERT_FRAME_THRESHOLD = 20
GESTURE_ALERT_COOLDOWN_SECONDS = 30