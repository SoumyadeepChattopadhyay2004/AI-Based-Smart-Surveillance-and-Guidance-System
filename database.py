"""
database.py
MongoDB Atlas data-access layer for users, attendance, alerts, movement,
mood and gesture logs. Function names/signatures match the original SQLite
version exactly, so app.py, face_utils.py etc. needed no changes.
"""

from datetime import datetime, timedelta

import config
from mongo import db as _db, next_id


def _out(doc):
    """Mongo's _id -> id, matching the dict shape templates/JS expect."""
    if doc is None:
        return None
    doc = dict(doc)
    doc["id"] = doc.pop("_id")
    return doc


def init_db():
    """No schema to create in MongoDB — just ensure the indexes we rely on exist."""
    _db.users.create_index("unique_id", unique=True)
    _db.attendance.create_index([("user_id", 1), ("date", 1)])
    _db.attendance.create_index("date")
    _db.alerts.create_index("timestamp")
    _db.movement_logs.create_index("timestamp")
    _db.mood_logs.create_index("timestamp")
    _db.gesture_logs.create_index("timestamp")


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------
def add_user(name, unique_id, email="", role="member"):
    if _db.users.find_one({"unique_id": unique_id}):
        raise ValueError(f"unique_id '{unique_id}' already exists")
    user_id = next_id("users")
    _db.users.insert_one({
        "_id": user_id,
        "name": name,
        "unique_id": unique_id,
        "email": email,
        "role": role,
        "created_at": datetime.now().isoformat(),
    })
    return user_id


def get_user_by_label(label_id):
    return _out(_db.users.find_one({"_id": label_id}))


def get_all_users():
    return [_out(d) for d in _db.users.find().sort("name", 1)]


def delete_user(user_id):
    _db.users.delete_one({"_id": user_id})


# ---------------------------------------------------------------------------
# Attendance
# ---------------------------------------------------------------------------
def mark_attendance(user_id, confidence):
    """Marks attendance, respecting a cooldown so the same face doesn't spam entries."""
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")

    last = next(_db.attendance.find({"user_id": user_id, "date": today}).sort("_id", -1).limit(1), None)

    if last:
        last_dt = datetime.strptime(f"{today} {last['time_in']}", "%Y-%m-%d %H:%M:%S")
        if datetime.now() - last_dt < timedelta(minutes=config.ATTENDANCE_COOLDOWN_MINUTES):
            return False  # too soon, skip duplicate

    _db.attendance.insert_one({
        "_id": next_id("attendance"),
        "user_id": user_id,
        "date": today,
        "time_in": now_time,
        "confidence": confidence,
        "status": "present",
    })
    return True


def get_attendance_for_date(date_str):
    rows = list(_db.attendance.find({"date": date_str}).sort("time_in", 1))
    users_by_id = {u["_id"]: u for u in _db.users.find({"_id": {"$in": [r["user_id"] for r in rows]}})}
    out = []
    for r in rows:
        row = _out(r)
        u = users_by_id.get(r["user_id"])
        row["name"] = u["name"] if u else None
        row["unique_id"] = u["unique_id"] if u else None
        out.append(row)
    return out


def get_attendance_summary(days=7):
    """Returns per-day present counts for the last N days, for charting."""
    pipeline = [
        {"$group": {"_id": "$date", "present_count": {"$addToSet": "$user_id"}}},
        {"$project": {"date": "$_id", "present_count": {"$size": "$present_count"}, "_id": 0}},
        {"$sort": {"date": -1}},
        {"$limit": days},
    ]
    rows = list(_db.attendance.aggregate(pipeline))
    return rows[::-1]


def get_user_attendance_rate():
    counts = {}
    for r in _db.attendance.find({}, {"user_id": 1}):
        counts[r["user_id"]] = counts.get(r["user_id"], 0) + 1

    out = []
    for u in _db.users.find().sort("name", 1):
        out.append({"name": u["name"], "days_present": counts.get(u["_id"], 0)})
    out.sort(key=lambda r: r["days_present"], reverse=True)
    return out


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------
def add_alert(alert_type, severity, description, snapshot_path=""):
    alert_id = next_id("alerts")
    _db.alerts.insert_one({
        "_id": alert_id,
        "alert_type": alert_type,
        "severity": severity,
        "description": description,
        "snapshot_path": snapshot_path,
        "timestamp": datetime.now().isoformat(),
        "acknowledged": 0,
    })
    return alert_id


def get_recent_alerts(limit=50):
    return [_out(d) for d in _db.alerts.find().sort("timestamp", -1).limit(limit)]


def acknowledge_alert(alert_id):
    _db.alerts.update_one({"_id": alert_id}, {"$set": {"acknowledged": 1}})


def delete_alert(alert_id):
    _db.alerts.delete_one({"_id": alert_id})


def clear_alerts():
    return _db.alerts.delete_many({}).deleted_count


def get_alert_counts():
    rows = _db.alerts.aggregate([{"$group": {"_id": "$alert_type", "count": {"$sum": 1}}}])
    return {r["_id"]: r["count"] for r in rows}


# ---------------------------------------------------------------------------
# Movement logs
# ---------------------------------------------------------------------------
def add_movement_log(zone, area_ratio, description, severity):
    _db.movement_logs.insert_one({
        "_id": next_id("movement_logs"),
        "zone": zone,
        "area_ratio": area_ratio,
        "description": description,
        "severity": severity,
        "timestamp": datetime.now().isoformat(),
    })


def get_recent_movement_logs(limit=50):
    return [_out(d) for d in _db.movement_logs.find().sort("timestamp", -1).limit(limit)]


def delete_movement_log(log_id):
    _db.movement_logs.delete_one({"_id": log_id})


def clear_movement_logs():
    return _db.movement_logs.delete_many({}).deleted_count


# ---------------------------------------------------------------------------
# Mood / facial-expression logs
# ---------------------------------------------------------------------------
def add_mood_log(user_id, emotion, confidence):
    _db.mood_logs.insert_one({
        "_id": next_id("mood_logs"),
        "user_id": user_id,
        "emotion": emotion,
        "confidence": confidence,
        "timestamp": datetime.now().isoformat(),
    })


def get_recent_moods(limit=50):
    rows = list(_db.mood_logs.find().sort("timestamp", -1).limit(limit))
    user_ids = [r["user_id"] for r in rows if r.get("user_id") is not None]
    users_by_id = {u["_id"]: u["name"] for u in _db.users.find({"_id": {"$in": user_ids}})}
    out = []
    for r in rows:
        row = _out(r)
        row["user_name"] = users_by_id.get(r.get("user_id"))
        out.append(row)
    return out


def get_mood_counts(hours=24):
    """Mood breakdown over the last `hours`, for a dashboard pie/bar chart."""
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    rows = _db.mood_logs.aggregate([
        {"$match": {"timestamp": {"$gte": cutoff}}},
        {"$group": {"_id": "$emotion", "count": {"$sum": 1}}},
    ])
    return {r["_id"]: r["count"] for r in rows}


def delete_mood_log(log_id):
    _db.mood_logs.delete_one({"_id": log_id})


def clear_mood_logs():
    return _db.mood_logs.delete_many({}).deleted_count


# ---------------------------------------------------------------------------
# Gesture logs
# ---------------------------------------------------------------------------
def add_gesture_log(gesture):
    _db.gesture_logs.insert_one({
        "_id": next_id("gesture_logs"),
        "gesture": gesture,
        "timestamp": datetime.now().isoformat(),
    })


def get_recent_gestures(limit=50):
    return [_out(d) for d in _db.gesture_logs.find().sort("timestamp", -1).limit(limit)]


def get_gesture_counts(hours=24):
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    rows = _db.gesture_logs.aggregate([
        {"$match": {"timestamp": {"$gte": cutoff}}},
        {"$group": {"_id": "$gesture", "count": {"$sum": 1}}},
    ])
    return {r["_id"]: r["count"] for r in rows}


def delete_gesture_log(log_id):
    _db.gesture_logs.delete_one({"_id": log_id})


def clear_gesture_logs():
    return _db.gesture_logs.delete_many({}).deleted_count
