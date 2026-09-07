"""
alert_system.py
Turns raw detection events (unknown face, suspicious movement) into a short,
human-readable alert description using Groq's LLM API. Falls back to a
plain template if no Groq API key is configured or the call fails, so the
rest of the app never breaks because of the network/API.
"""

from datetime import datetime

import config

_client = None
if config.GROQ_ENABLED:
    try:
        from groq import Groq
        _client = Groq(api_key=config.GROQ_API_KEY)
    except Exception:
        _client = None


def _fallback_text(event_type, details):
    if event_type == "unknown_person":
        return (
            f"Unrecognized individual detected by camera at "
            f"{details.get('time', datetime.now().strftime('%H:%M:%S'))}. "
            f"Face did not match any record in the registered database. "
            f"Recommend visual verification."
        )
    if event_type == "suspicious_movement":
        return (
            f"Unusual/rapid movement detected covering "
            f"{details.get('area_ratio', 0) * 100:.1f}% of the camera frame "
            f"at {details.get('time', datetime.now().strftime('%H:%M:%S'))}. "
            f"Recommend review of live feed."
        )
    if event_type == "mood_detected":
        return (
            f"{details.get('emotion', 'Unusual')} mood sustained on "
            f"{details.get('person', 'a person')} at "
            f"{details.get('time', datetime.now().strftime('%H:%M:%S'))}. "
            f"Recommend visual check-in."
        )
    if event_type == "gesture_detected":
        return (
            f"'{details.get('gesture', 'A')}' gesture held toward the camera "
            f"at {details.get('time', datetime.now().strftime('%H:%M:%S'))}. "
            f"Recommend review of live feed for a possible signal."
        )
    return "Unclassified event detected."


def generate_alert_text(event_type, details):
    """
    event_type: 'unknown_person' | 'suspicious_movement'
    details: dict of raw signals, e.g. {"time": "...", "confidence": ..., "area_ratio": ...}
    Returns a short natural-language alert string suitable for a dashboard/notification.
    """
    if _client is None:
        return _fallback_text(event_type, details)

    prompt = (
        "You are the alerting module of a campus/office AI surveillance system. "
        "Write ONE short, clear, professional alert sentence (max 35 words) for a "
        "security dashboard based on this raw event data. Do not add greetings, "
        "explanations, or markdown — output only the alert sentence.\n\n"
        f"Event type: {event_type}\n"
        f"Details: {details}\n"
    )

    try:
        response = _client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.4,
        )
        text = response.choices[0].message.content.strip()
        return text if text else _fallback_text(event_type, details)
    except Exception:
        return _fallback_text(event_type, details)


def generate_daily_summary(attendance_rows, alert_rows):
    """
    Produces a short natural-language daily summary paragraph for the
    dashboard using Groq, given today's attendance and alert records.
    Falls back to a simple templated summary if Groq is unavailable.
    """
    present_count = len(attendance_rows)
    unknown_count = sum(1 for a in alert_rows if a["alert_type"] == "unknown_person")
    movement_count = sum(1 for a in alert_rows if a["alert_type"] == "suspicious_movement")

    if _client is None:
        return (
            f"Today, {present_count} people were marked present via facial "
            f"attendance. The system raised {unknown_count} unknown-person "
            f"alert(s) and {movement_count} suspicious-movement alert(s)."
        )

    prompt = (
        "You are a security operations assistant. Write a concise 2-3 sentence "
        "daily summary for a dashboard, in a neutral professional tone, based "
        "on this data. Output plain text only, no markdown.\n\n"
        f"People present today (via face recognition attendance): {present_count}\n"
        f"Unknown-person alerts: {unknown_count}\n"
        f"Suspicious-movement alerts: {movement_count}\n"
    )

    try:
        response = _client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.5,
        )
        text = response.choices[0].message.content.strip()
        return text or f"{present_count} present, {unknown_count} unknown-person alerts, {movement_count} movement alerts today."
    except Exception:
        return (
            f"Today, {present_count} people were marked present via facial "
            f"attendance. The system raised {unknown_count} unknown-person "
            f"alert(s) and {movement_count} suspicious-movement alert(s)."
        )