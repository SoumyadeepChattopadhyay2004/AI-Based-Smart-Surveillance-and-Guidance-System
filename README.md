# SENTRY — AI-Based Smart Surveillance and Guidance System

A Flask + OpenCV surveillance and attendance system with a themed web dashboard.

## Features (mapped to code)

| Feature | Where it lives |
|---|---|
| Attractive GUI with backend + database | `templates/`, `static/css/style.css`, Flask (`app.py`), SQLite (`database.py`) |
| Facial-scan based attendance | `face_utils.py` + `/live_feed` route in `app.py` |
| Personal dashboard & analytics | `/dashboard`, `templates/dashboard.html`, Chart.js, `/api/*` routes |
| Face detection & recognition | `face_utils.py` (Haar Cascade detection + OpenCV LBPH recognition) |
| Unknown person identification & alerts | `/live_feed` unknown-streak logic + `alert_system.py` (Groq API) |
| Suspicious movement tracker | `motion_detector.py` (MOG2 background subtraction) |
| Mood / facial-expression detection | `emotion_utils.py` (ONNX FER+ model via `cv2.dnn`) + `/live_feed` overlay + `mood_logs` table |
| Hand gesture detection | `gesture_utils.py` (MediaPipe Hands) + `/live_feed` overlay + `gesture_logs` table |

## How recognition works

This project intentionally uses **OpenCV's built-in `cv2.face.LBPHFaceRecognizer`**
instead of `dlib` / `face_recognition`, because LBPH has zero compiled
dependencies beyond `opencv-contrib-python` — it installs cleanly everywhere
(Windows, Mac, Linux, low-spec machines) with a single `pip install`, which
matters a lot for a student/prototype project. Detection uses the classic
Haar Cascade frontal-face model that ships with OpenCV.

## AI Alerts (Groq API)

Instead of showing raw event flags, `alert_system.py` sends event data
(unknown face detected / suspicious movement %) to **Groq's LLM API**
(`llama-3.3-70b-versatile` by default) to generate a short, readable alert
sentence for the dashboard, plus a daily natural-language summary. If no
API key is configured, the app **automatically falls back** to clean
template-based text — the system always runs end-to-end even without Groq.

## Mood detection (facial expressions)

`emotion_utils.py` classifies each detected face into one of 8 moods —
**Happy, Sad, Angry, Surprised, Anxious, Disgusted, Neutral, Contempt** —
using a small ONNX "FER+" CNN run through `cv2.dnn`, so no extra ML
framework (PyTorch/TensorFlow) is required, staying consistent with the
OpenCV-only philosophy above.

One-time setup — download the model and save it at `models/emotion-ferplus-8.onnx`:

```bash
mkdir -p models
curl -L -o models/emotion-ferplus-8.onnx \
  https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx
```

If the file isn't present, mood detection quietly disables itself (the rest
of the app is unaffected). Detected moods are logged to `mood_logs`, shown
under each face in `/live_feed`, summarized on the dashboard, and — if a
face shows `Angry` or `Anxious` continuously (`MOOD_ALERT_FRAME_THRESHOLD`
frames) — raise a `mood_detected` alert like the other alert types.

## Gesture detection

`gesture_utils.py` recognizes a small set of hand gestures — **Open Palm,
Fist, Thumbs Up, Pointing, Peace/V Sign, Call Me** — using MediaPipe Hands
landmark detection plus a simple rule-based classifier over which fingers
are extended. No model download is needed beyond `pip install mediapipe`
(already in `requirements.txt`); MediaPipe ships its own hand-landmark
model internally.

Every detected gesture is logged to `gesture_logs` and drawn as a hand
skeleton + label on `/live_feed`. Holding an **Open Palm** toward the
camera for a sustained number of frames (`GESTURE_ALERT_GESTURES` /
`GESTURE_ALERT_FRAME_THRESHOLD` in `config.py`) — e.g. as a stop/help
signal — raises a `gesture_detected` alert.

Both features are toggled independently via `config.ENABLE_EMOTION_DETECTION`
and `config.ENABLE_GESTURE_DETECTION`, and both fail open: if a model/package
isn't set up, the live feed keeps running with whatever else is available.

> **Templates:** this repo's `templates/*.html` files weren't part of the
> files edited here. The backend now passes `mood_counts`, `gesture_counts`,
> `recent_moods`, `mood_logs`, `gesture_logs`, `emotion_ready`, and
> `gesture_ready` into `dashboard.html`, `live_monitor.html`, and
> `alerts.html` — add matching Jinja markup (e.g. a mood pie chart, a
> gesture log table) wherever you'd like them displayed.

## Storage: MongoDB Atlas (no local database)

All data — users, attendance, alerts, movement/mood/gesture logs, face
sample images, the trained LBPH model, and alert snapshots — is stored in
MongoDB Atlas. Nothing is written to local disk except brief temp files
that OpenCV's file-based model API forces (deleted immediately after use).
See `mongo.py` (connection + GridFS bucket), `storage.py` (binary files),
and `database.py` (structured records).

1. Create a free cluster at https://www.mongodb.com/cloud/atlas/register
2. Under **Database Access**, create a user with a password.
3. Under **Network Access**, allow your IP (or `0.0.0.0/0` for a quick student setup).
4. Copy your connection string (Atlas UI → **Connect** → **Drivers**), it looks like:
   `mongodb+srv://<user>:<password>@<cluster>.mongodb.net/`

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your MongoDB Atlas connection string
export MONGODB_URI="mongodb+srv://<user>:<password>@<cluster>.mongodb.net/"   # Windows: set MONGODB_URI=...

# 4. (Optional but recommended) Set your Groq API key
export GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxx"     # Windows: set GROQ_API_KEY=...
# Get a free key at https://console.groq.com/keys

# 5. Mood detection model is already bundled at models/emotion-ferplus-8.onnx
#    (this one static asset stays local; it's not user data)

# 6. Run the app
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

> Requires a webcam. If you have more than one camera, change
> `CAMERA_INDEX` in `config.py` (0, 1, 2 …).

## Using the system

1. **Enroll Person** → fill in name + unique ID → camera opens and captures
   30 face samples automatically → the LBPH model retrains itself.
2. **Live Monitor** → open the live surveillance feed. Recognized faces are
   boxed in teal and marked present in Attendance; unrecognized faces are
   boxed in red and, after a sustained detection, raise an "Unknown Person"
   alert. Large/rapid movement in the frame is boxed in amber and raises a
   "Suspicious Movement" alert. Each detected face also shows its mood
   (Happy/Sad/Angry/Surprised/Anxious/Disgusted/Neutral/Contempt) if the
   mood model is set up, and any hand gestures in frame (Open Palm, Fist,
   Thumbs Up, Pointing, Peace/V Sign, Call Me) are skeleton-outlined and
   labeled if MediaPipe is installed.
3. **Attendance Log** → filter attendance by date.
4. **Alerts** → see all AI-generated alert descriptions and the raw
   movement log; acknowledge alerts as they're reviewed.
5. **Dashboard** → present-today count, weekly attendance trend, alert
   breakdown chart, and an AI-generated daily summary.

## Project structure

```
smart_surveillance_system/
├── app.py                # Flask routes + video streaming (MJPEG)
├── config.py              # All settings incl. Groq API key/model, MongoDB URI
├── mongo.py                # MongoDB Atlas connection + GridFS bucket (shared)
├── storage.py               # GridFS: face samples, trained model, alert snapshots
├── database.py                # Atlas queries (users/attendance/alerts/movement/mood/gesture)
├── face_utils.py                # Haar detection + LBPH training/recognition (via storage.py)
├── motion_detector.py             # Background-subtraction suspicious movement detector
├── alert_system.py                 # Groq-powered alert text + daily summary (with fallback)
├── emotion_utils.py                  # Mood detection — ONNX FER+ model via cv2.dnn (with fallback)
├── gesture_utils.py                    # Hand gesture recognition — MediaPipe Hands (with fallback)
├── requirements.txt
├── templates/                            # Jinja2 HTML (dashboard, live monitor, attendance, alerts, enrollment)
├── static/css/style.css                    # Dark "control room" themed UI
└── models/emotion-ferplus-8.onnx             # Mood detection model (bundled static asset, stays local)
```

Face samples, the trained model + label map, alert snapshots, and all
database records live in MongoDB Atlas — no `known_faces/`, `trainer/`,
`alert_snapshots/`, or `database/` folders are used anymore.

## Notes for a report / viva

- **Database**: SQLite, 4 tables — `users`, `attendance`, `alerts`,
  `movement_logs` (see `database.py` for the full schema and indices).
- **Recognition threshold**: `LBPH_CONFIDENCE_THRESHOLD` in `config.py`
  controls the sensitivity of "known vs unknown" — lower distance =
  more confident match; tune this per lighting/hardware.
- **Attendance de-duplication**: a per-user, per-day cooldown
  (`ATTENDANCE_COOLDOWN_MINUTES`) stops the same face marking multiple
  entries while lingering in frame.
- **Alert de-bouncing**: both the unknown-person and movement detectors use
  frame-streak / time-cooldown logic so a single event doesn't flood the
  alerts table.
- This is a functional prototype meant for a controlled setting (classroom,
  lab, office desk) with a single camera — it is not a production-grade
  multi-camera security system.