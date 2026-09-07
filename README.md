<div align="center">

# 🛡️ SENTRY
### AI-Based Smart Surveillance & Guidance System

**A single-camera, AI-powered monitoring system for classrooms, labs, and offices —**
**face-recognition attendance, live threat detection, mood & gesture awareness, and LLM-generated alerts, wrapped in a dark "control room" dashboard.**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Backend-black?logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-Vision-5C3EE8?logo=opencv&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?logo=mongodb&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLM%20Alerts-orange)
![License](https://img.shields.io/badge/status-student%20prototype-yellow)

</div>

---

## ✨ Overview

SENTRY watches a single webcam feed and turns raw video into a running story of what's happening in the room: **who's present**, **who isn't recognized**, **whether something moved unusually**, **how people are feeling**, and **what they're gesturing** — then reports all of it in plain language on a live dashboard.

It's built as a **student/prototype project with zero heavyweight dependencies**: no `dlib`, no PyTorch/TensorFlow, no local database file. Everything runs on OpenCV + MediaPipe + a small ONNX model, and every record lives in MongoDB Atlas.

## 🧩 Features at a Glance

| Capability | How it works | Where |
|---|---|---|
| 🎥 **Live surveillance dashboard** | Flask + MJPEG stream, dark themed UI | `app.py`, `templates/`, `static/css/style.css` |
| 🧑‍🤝‍🧑 **Face-recognition attendance** | Haar Cascade detection + OpenCV LBPH recognition | `face_utils.py` |
| 🚨 **Unknown-visitor alerts** | Sustained unrecognized-face streak → alert | `app.py` (`/live_feed`) + `alert_system.py` |
| 🏃 **Suspicious movement detection** | MOG2 background subtraction | `motion_detector.py` |
| 🙂 **Mood / expression detection** | ONNX FER+ model via `cv2.dnn` (8 emotions) | `emotion_utils.py` |
| ✋ **Hand gesture recognition** | MediaPipe Hands + rule-based classifier | `gesture_utils.py` |
| 🤖 **Natural-language alerts** | Groq LLM turns raw events into readable text | `alert_system.py` |
| ☁️ **Cloud storage, zero local DB** | Users, logs, face samples, trained model, snapshots | `mongo.py`, `storage.py`, `database.py` |

---

## 🧠 Why These Choices

**Face recognition — LBPH, not `dlib`.**
`cv2.face.LBPHFaceRecognizer` has no compiled dependencies beyond `opencv-contrib-python`, so it installs cleanly on Windows/Mac/Linux and low-spec machines with a single `pip install`. Detection uses OpenCV's classic Haar Cascade frontal-face model.

**Mood detection — a tiny ONNX model, not a full ML framework.**
`emotion_utils.py` classifies faces into **Happy, Sad, Angry, Surprised, Anxious, Disgusted, Neutral, Contempt** using a small "FER+" CNN run entirely through `cv2.dnn` — no PyTorch/TensorFlow required, staying consistent with the OpenCV-only philosophy.

**Gesture detection — MediaPipe Hands.**
Recognizes **Open Palm, Fist, Thumbs Up, Pointing, Peace/V Sign, Call Me** via landmark detection and a simple rule-based classifier over which fingers are extended. No model download needed — MediaPipe ships its own.

**Alerts — Groq LLM with a safe fallback.**
Instead of raw event flags, `alert_system.py` sends event data to Groq (`llama-3.3-70b-versatile` by default) to generate short, human-readable alert sentences and a daily summary. **No API key? No problem** — it automatically falls back to clean template-based text, so the system always runs end-to-end.

> Both mood and gesture detection are independently toggleable (`config.ENABLE_EMOTION_DETECTION`, `config.ENABLE_GESTURE_DETECTION`) and **fail open** — if a model or package isn't set up, the live feed keeps running with whatever else is available.

---

## ☁️ Storage: MongoDB Atlas (no local database)

Everything — users, attendance, alerts, movement/mood/gesture logs, face samples, the trained LBPH model, and alert snapshots — lives in MongoDB Atlas. Nothing touches local disk except brief temp files OpenCV's file-based model API requires, deleted immediately after use.

| File | Responsibility |
|---|---|
| `mongo.py` | Atlas connection + shared GridFS bucket |
| `storage.py` | Binary files — face samples, trained model, snapshots |
| `database.py` | Structured records — users, attendance, alerts, logs |

**Setting up your cluster:**

1. Create a free cluster → [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register)
2. **Database Access** → create a user with a password
3. **Network Access** → allow your IP (or `0.0.0.0/0` for quick student setup)
4. **Connect → Drivers** → copy your connection string:
   ```
   mongodb+srv://<user>:<password>@<cluster>.mongodb.net/
   ```
   ⚠️ Replace `<user>`, `<password>`, and `<cluster>` with your **actual** values — a literal `<cluster>` left in the string is the #1 cause of the `DNS query name does not exist` error on startup.

---

## 🚀 Setup

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
# Free key: https://console.groq.com/keys

# 5. Mood detection model is already bundled at models/emotion-ferplus-8.onnx
#    (this one static asset stays local — it's not user data)

# 6. Run the app
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

> 📷 Requires a webcam. Multiple cameras? Change `CAMERA_INDEX` in `config.py` (0, 1, 2 …).

---

## 🕹️ Using the System

1. **Enroll Person** — enter name + unique ID → camera captures 30 face samples automatically → LBPH model retrains itself.
2. **Live Monitor** — watch the feed live:
   - 🟦 Teal box → recognized face, marked present
   - 🟥 Red box → unrecognized face → "Unknown Person" alert after a sustained detection
   - 🟨 Amber box → large/rapid movement → "Suspicious Movement" alert
   - Mood label under each face (if the model is set up)
   - Gesture skeleton + label in-frame (if MediaPipe is installed)
3. **Attendance Log** — filter attendance by date.
4. **Alerts** — review AI-generated alert descriptions and the raw movement log; acknowledge as reviewed.
5. **Dashboard** — present-today count, weekly attendance trend, alert breakdown chart, and an AI-generated daily summary.

---

## 🗂️ Project Structure

```
smart_surveillance_system/
├── app.py                  # Flask routes + video streaming (MJPEG)
├── config.py                # All settings: Groq key/model, MongoDB URI, thresholds
├── mongo.py                  # MongoDB Atlas connection + shared GridFS bucket
├── storage.py                  # GridFS: face samples, trained model, alert snapshots
├── database.py                   # Atlas queries: users / attendance / alerts / logs
├── face_utils.py                   # Haar detection + LBPH training/recognition
├── motion_detector.py                # Background-subtraction movement detector
├── alert_system.py                     # Groq-powered alert text + daily summary
├── emotion_utils.py                      # Mood detection — ONNX FER+ via cv2.dnn
├── gesture_utils.py                        # Hand gesture recognition — MediaPipe
├── requirements.txt
├── templates/                                # Jinja2 HTML views
├── static/css/style.css                        # Dark "control room" theme
└── models/emotion-ferplus-8.onnx                 # Bundled mood-detection model
```

*Face samples, the trained model, snapshots, and all records live in Atlas — no `known_faces/`, `trainer/`, `alert_snapshots/`, or `database/` folders needed.*

---

## 📝 Notes for a Report / Viva

- **Recognition threshold** — `LBPH_CONFIDENCE_THRESHOLD` in `config.py` controls known-vs-unknown sensitivity (lower distance = more confident match); tune per lighting/hardware.
- **Attendance de-duplication** — a per-user, per-day cooldown (`ATTENDANCE_COOLDOWN_MINUTES`) prevents duplicate entries while a face lingers in frame.
- **Alert de-bouncing** — unknown-person and movement detectors use frame-streak / time-cooldown logic so a single event doesn't flood the alerts table.
- **Scope** — this is a functional prototype for a controlled single-camera setting (classroom, lab, office desk), not a production-grade multi-camera security system.

---

<div align="center">

Built as a single-camera AI prototype — contributions and forks welcome.

</div>
