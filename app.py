"""
app.py
AI-Based Smart Surveillance and Guidance System — Flask backend.

Run with:  python app.py
Then open: http://127.0.0.1:5000
"""

import os
import time
import json
from datetime import datetime

import cv2
from flask import (
    Flask, render_template, request, redirect, url_for,
    Response, jsonify, flash, session
)

import config
import database as db
import face_utils
import storage
import auth
import alert_system
from motion_detector import MotionDetector
from emotion_utils import EmotionDetector, EMOTION_COLORS
from gesture_utils import GestureDetector

app = Flask(__name__)
app.secret_key = config.SECRET_KEY

db.init_db()
recognizer = face_utils.FaceRecognizer()
motion_detector = MotionDetector()
emotion_detector = EmotionDetector()   # no-ops until models/emotion-ferplus-8.onnx exists
gesture_detector = GestureDetector()   # no-ops if mediapipe isn't installed

# Simple in-process camera handle shared by the streaming routes.
_camera = None


def get_camera():
    global _camera
    if _camera is None or not _camera.isOpened():
        _camera = cv2.VideoCapture(config.CAMERA_INDEX)
        _camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        _camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return _camera


def release_camera():
    global _camera
    if _camera is not None:
        _camera.release()
        _camera = None


# ---------------------------------------------------------------------------
# Auth gate — login is mandatory for every route except the public ones below
# ---------------------------------------------------------------------------
PUBLIC_ENDPOINTS = {"landing", "login", "signup", "logout", "static"}


@app.before_request
def _require_login():
    if request.endpoint in PUBLIC_ENDPOINTS or request.endpoint is None:
        return
    if not session.get("account_id"):
        return redirect(url_for("login", next=request.path))


# ---------------------------------------------------------------------------
# Public pages — landing + auth
# ---------------------------------------------------------------------------
@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        next_url = request.form.get("next") or url_for("dashboard")

        account = auth.verify_account(username, password)
        if not account:
            flash("Invalid username or password.", "error")
            return redirect(url_for("login", next=next_url))

        session["account_id"] = account["id"]
        session["username"] = account["username"]
        return redirect(next_url)

    return render_template("login.html", next=request.args.get("next", ""))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        if not username or not password:
            flash("Username and password are required.", "error")
            return redirect(url_for("signup"))
        if password != confirm:
            flash("Passwords do not match.", "error")
            return redirect(url_for("signup"))

        try:
            account_id = auth.create_account(username, password, email)
        except ValueError as e:
            flash(str(e), "error")
            return redirect(url_for("signup"))

        session["account_id"] = account_id
        session["username"] = username
        flash("Account created — welcome!", "success")
        return redirect(url_for("dashboard"))

    return render_template("signup.html")


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("landing"))


# ---------------------------------------------------------------------------
# Dashboard pages (login required — enforced by _require_login above)
# ---------------------------------------------------------------------------


@app.route("/dashboard")
def dashboard():
    today = datetime.now().strftime("%Y-%m-%d")
    today_attendance = db.get_attendance_for_date(today)
    total_users = len(db.get_all_users())
    recent_alerts = db.get_recent_alerts(limit=8)
    alert_counts = db.get_alert_counts()
    weekly_summary = db.get_attendance_summary(days=7)
    mood_counts = db.get_mood_counts(hours=24)
    gesture_counts = db.get_gesture_counts(hours=24)
    recent_moods = db.get_recent_moods(limit=8)

    summary_text = alert_system.generate_daily_summary(today_attendance, db.get_recent_alerts(limit=200))

    return render_template(
        "dashboard.html",
        today_attendance=today_attendance,
        total_users=total_users,
        present_today=len(today_attendance),
        recent_alerts=recent_alerts,
        alert_counts=alert_counts,
        weekly_summary=weekly_summary,
        summary_text=summary_text,
        groq_enabled=config.GROQ_ENABLED,
        mood_counts=mood_counts,
        gesture_counts=gesture_counts,
        recent_moods=recent_moods,
        emotion_ready=emotion_detector.is_ready,
        gesture_ready=gesture_detector.is_ready,
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        unique_id = request.form.get("unique_id", "").strip()
        email = request.form.get("email", "").strip()

        if not name or not unique_id:
            flash("Name and Unique ID are required.", "error")
            return redirect(url_for("register"))

        try:
            user_id = db.add_user(name, unique_id, email)
        except Exception as e:
            flash(f"Could not register user (is the ID already taken?): {e}", "error")
            return redirect(url_for("register"))

        return redirect(url_for("capture_samples", user_id=user_id))

    return render_template("register.html")


@app.route("/capture/<int:user_id>")
def capture_samples(user_id):
    user = db.get_user_by_label(user_id)
    if not user:
        flash("User not found.", "error")
        return redirect(url_for("register"))
    return render_template("capture.html", user=user, samples_needed=config.FACE_SAMPLES_PER_USER)


@app.route("/capture_feed/<int:user_id>")
def capture_feed(user_id):
    """MJPEG stream that captures face samples live and shows progress overlay."""

    def generate():
        cam = get_camera()
        count = storage.count_face_samples(user_id)
        while count < config.FACE_SAMPLES_PER_USER:
            ok, frame = cam.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_utils.detect_faces(gray)
            for (x, y, w, h) in faces:
                face_img = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
                ok2, sample_buf = cv2.imencode(".jpg", face_img)
                if ok2:
                    storage.save_face_sample(user_id, count, sample_buf.tobytes())
                    count += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 217, 192), 2)
                break

            cv2.putText(
                frame, f"Samples: {count}/{config.FACE_SAMPLES_PER_USER}",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 217, 192), 2,
            )
            ok2, buffer = cv2.imencode(".jpg", frame)
            if not ok2:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

        # Training happens once enough samples exist
        face_utils.train_recognizer()
        recognizer.reload()

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/train", methods=["POST"])
def retrain():
    count = face_utils.train_recognizer()
    recognizer.reload()
    flash(f"Recognizer retrained on {count} user(s).", "success")
    return redirect(url_for("users_page"))


@app.route("/users")
def users_page():
    return render_template("users.html", users=db.get_all_users())


@app.route("/users/delete/<int:user_id>", methods=["POST"])
def delete_user(user_id):
    db.delete_user(user_id)
    storage.delete_face_samples(user_id)
    face_utils.train_recognizer()
    recognizer.reload()
    flash("User removed.", "success")
    return redirect(url_for("users_page"))


@app.route("/live_monitor")
def live_monitor():
    return render_template(
        "live_monitor.html",
        recognizer_ready=recognizer.is_ready,
        emotion_ready=emotion_detector.is_ready,
        gesture_ready=gesture_detector.is_ready,
    )


@app.route("/live_feed")
def live_feed():
    """
    Main MJPEG surveillance stream: runs face detection + recognition
    (marks attendance / raises unknown-person alerts) and suspicious
    movement detection simultaneously, overlaying results on the video.
    """
    def generate():
        cam = get_camera()
        unknown_streak = 0
        last_unknown_alert = 0.0

        mood_streaks = {}          # emotion label -> consecutive-frame count
        last_mood_alert = 0.0
        last_mood_log_by_user = {}  # user_id (or "unknown") -> last DB-write time

        gesture_streaks = {}       # gesture label -> consecutive-frame count
        last_gesture_alert = 0.0

        while True:
            ok, frame = cam.read()
            if not ok:
                time.sleep(0.05)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_utils.detect_faces(gray)

            any_unknown_this_frame = False

            for (x, y, w, h) in faces:
                face_img = gray[y:y + h, x:x + w]
                user_id, confidence = recognizer.predict(face_img)

                if user_id is not None:
                    user = db.get_user_by_label(user_id)
                    label = user["name"] if user else "Unknown"
                    color = (0, 217, 192)  # teal = recognized
                    if user:
                        db.mark_attendance(user_id, confidence)
                else:
                    label = "UNKNOWN"
                    color = (68, 68, 255)  # red = unknown
                    any_unknown_this_frame = True

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # --- Mood / facial-expression detection ---
                if config.ENABLE_EMOTION_DETECTION and emotion_detector.is_ready:
                    mood, mood_conf = emotion_detector.detect_mood(face_img)
                    if mood != "Unknown":
                        mood_color = EMOTION_COLORS.get(mood, (200, 200, 200))
                        cv2.putText(frame, f"{mood} ({mood_conf * 100:.0f}%)", (x, y + h + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, mood_color, 2)

                        log_key = user_id if user_id is not None else "unknown"
                        now_t = time.time()
                        if now_t - last_mood_log_by_user.get(log_key, 0) >= config.MOOD_LOG_INTERVAL_SECONDS:
                            last_mood_log_by_user[log_key] = now_t
                            db.add_mood_log(user_id, mood, mood_conf)

                        if mood in config.MOOD_ALERT_EMOTIONS:
                            mood_streaks[mood] = mood_streaks.get(mood, 0) + 1
                        else:
                            mood_streaks[mood] = 0

                        if (mood_streaks.get(mood, 0) >= config.MOOD_ALERT_FRAME_THRESHOLD
                                and (now_t - last_mood_alert) >= config.MOOD_ALERT_COOLDOWN_SECONDS):
                            last_mood_alert = now_t
                            snapshot_path = _save_snapshot(frame, "mood")
                            text = alert_system.generate_alert_text(
                                "mood_detected",
                                {"time": datetime.now().strftime("%H:%M:%S"), "emotion": mood, "person": label},
                            )
                            db.add_alert("mood_detected", "medium", text, snapshot_path)

            # --- Hand gesture detection (whole frame, not tied to a face) ---
            if config.ENABLE_GESTURE_DETECTION and gesture_detector.is_ready:
                gestures = gesture_detector.detect(frame)
                seen_this_frame = set()
                for hand in gestures:
                    gx, gy, gw, gh = hand["bbox"]
                    gesture_detector.draw_landmarks(frame, hand["landmarks"])
                    cv2.putText(frame, hand["gesture"], (gx, max(gy - 10, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 217, 192), 2)
                    db.add_gesture_log(hand["gesture"])
                    seen_this_frame.add(hand["gesture"])

                for g in config.GESTURE_ALERT_GESTURES:
                    if g in seen_this_frame:
                        gesture_streaks[g] = gesture_streaks.get(g, 0) + 1
                    else:
                        gesture_streaks[g] = 0

                    now_t = time.time()
                    if (gesture_streaks.get(g, 0) >= config.GESTURE_ALERT_FRAME_THRESHOLD
                            and (now_t - last_gesture_alert) >= config.GESTURE_ALERT_COOLDOWN_SECONDS):
                        last_gesture_alert = now_t
                        snapshot_path = _save_snapshot(frame, "gesture")
                        text = alert_system.generate_alert_text(
                            "gesture_detected",
                            {"time": datetime.now().strftime("%H:%M:%S"), "gesture": g},
                        )
                        db.add_alert("gesture_detected", "medium", text, snapshot_path)

            # --- Unknown person alert logic ---
            if any_unknown_this_frame:
                unknown_streak += 1
            else:
                unknown_streak = 0

            now = time.time()
            if (unknown_streak >= config.UNKNOWN_FACE_ALERT_FRAME_THRESHOLD
                    and (now - last_unknown_alert) >= config.UNKNOWN_ALERT_COOLDOWN_SECONDS):
                last_unknown_alert = now
                snapshot_path = _save_snapshot(frame, "unknown")
                text = alert_system.generate_alert_text(
                    "unknown_person",
                    {"time": datetime.now().strftime("%H:%M:%S")},
                )
                db.add_alert("unknown_person", "high", text, snapshot_path)

            # --- Suspicious movement detection ---
            motion = motion_detector.analyze(frame)
            for (mx, my, mw, mh) in motion["moving_boxes"]:
                box_color = (0, 184, 255) if motion["is_suspicious"] else (120, 120, 120)
                cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), box_color, 1)

            if motion["is_suspicious"]:
                cv2.putText(frame, "SUSPICIOUS MOVEMENT", (15, frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 184, 255), 2)

            if motion["can_alert"]:
                snapshot_path = _save_snapshot(frame, "movement")
                text = alert_system.generate_alert_text(
                    "suspicious_movement",
                    {"time": datetime.now().strftime("%H:%M:%S"), "area_ratio": motion["area_ratio"]},
                )
                db.add_alert("suspicious_movement", "medium", text, snapshot_path)
                db.add_movement_log("main_camera", motion["area_ratio"], text, "medium")

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (15, frame.shape[0] - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            ok2, buffer = cv2.imencode(".jpg", frame)
            if not ok2:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


def _save_snapshot(frame, prefix):
    fname = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        return ""
    return storage.save_alert_snapshot(buffer.tobytes(), fname)  # GridFS file id


@app.route("/snapshot/<file_id>")
def serve_snapshot(file_id):
    """Serves an alert snapshot straight out of GridFS (used by alerts.html)."""
    try:
        data = storage.load_snapshot_bytes(file_id)
    except Exception:
        return "", 404
    return Response(data, mimetype="image/jpeg")


@app.route("/stop_camera", methods=["POST"])
def stop_camera():
    release_camera()
    return jsonify({"status": "camera released"})


@app.route("/attendance")
def attendance_page():
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    rows = db.get_attendance_for_date(date_str)
    return render_template("attendance.html", rows=rows, date_str=date_str)


@app.route("/alerts")
def alerts_page():
    alerts = db.get_recent_alerts(limit=100)
    movement_logs = db.get_recent_movement_logs(limit=50)
    mood_logs = db.get_recent_moods(limit=50)
    gesture_logs = db.get_recent_gestures(limit=50)
    return render_template(
        "alerts.html",
        alerts=alerts,
        movement_logs=movement_logs,
        mood_logs=mood_logs,
        gesture_logs=gesture_logs,
    )


@app.route("/alerts/ack/<int:alert_id>", methods=["POST"])
def ack_alert(alert_id):
    db.acknowledge_alert(alert_id)
    return jsonify({"status": "ok"})


@app.route("/alerts/delete/<int:alert_id>", methods=["POST"])
def delete_alert(alert_id):
    db.delete_alert(alert_id)
    return jsonify({"status": "ok"})


@app.route("/alerts/clear", methods=["POST"])
def clear_alerts():
    count = db.clear_alerts()
    return jsonify({"status": "ok", "count": count})


@app.route("/movement_logs/delete/<int:log_id>", methods=["POST"])
def delete_movement_log(log_id):
    db.delete_movement_log(log_id)
    return jsonify({"status": "ok"})


@app.route("/movement_logs/clear", methods=["POST"])
def clear_movement_logs():
    count = db.clear_movement_logs()
    return jsonify({"status": "ok", "count": count})


@app.route("/mood_logs/delete/<int:log_id>", methods=["POST"])
def delete_mood_log(log_id):
    db.delete_mood_log(log_id)
    return jsonify({"status": "ok"})


@app.route("/mood_logs/clear", methods=["POST"])
def clear_mood_logs():
    count = db.clear_mood_logs()
    return jsonify({"status": "ok", "count": count})


@app.route("/gesture_logs/delete/<int:log_id>", methods=["POST"])
def delete_gesture_log(log_id):
    db.delete_gesture_log(log_id)
    return jsonify({"status": "ok"})


@app.route("/gesture_logs/clear", methods=["POST"])
def clear_gesture_logs():
    count = db.clear_gesture_logs()
    return jsonify({"status": "ok", "count": count})


# ---------------------------------------------------------------------------
# JSON API for dashboard charts (Chart.js)
# ---------------------------------------------------------------------------
@app.route("/api/weekly_attendance")
def api_weekly_attendance():
    return jsonify(db.get_attendance_summary(days=7))


@app.route("/api/attendance_rate")
def api_attendance_rate():
    return jsonify(db.get_user_attendance_rate())


@app.route("/api/alert_counts")
def api_alert_counts():
    return jsonify(db.get_alert_counts())


@app.route("/api/mood_counts")
def api_mood_counts():
    hours = request.args.get("hours", default=24, type=int)
    return jsonify(db.get_mood_counts(hours=hours))


@app.route("/api/gesture_counts")
def api_gesture_counts():
    hours = request.args.get("hours", default=24, type=int)
    return jsonify(db.get_gesture_counts(hours=hours))


if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, threaded=True)