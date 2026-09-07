"""
auth.py
Session-based login for accessing the SENTRY dashboard itself — distinct
from the face-recognition "users" collection (people enrolled for
attendance/recognition). Accounts live in MongoDB Atlas (`accounts`
collection); passwords are hashed with Werkzeug's
generate_password_hash/check_password_hash — plaintext is never stored.
"""

from datetime import datetime

from werkzeug.security import generate_password_hash, check_password_hash

from mongo import db as _db, next_id


def create_account(username, password, email=""):
    if _db.accounts.find_one({"username": username}):
        raise ValueError(f"Username '{username}' is already taken.")
    account_id = next_id("accounts")
    _db.accounts.insert_one({
        "_id": account_id,
        "username": username,
        "password_hash": generate_password_hash(password),
        "email": email,
        "created_at": datetime.now().isoformat(),
    })
    return account_id


def verify_account(username, password):
    """Returns {id, username, email} on success, else None."""
    acc = _db.accounts.find_one({"username": username})
    if not acc or not check_password_hash(acc["password_hash"], password):
        return None
    return {"id": acc["_id"], "username": acc["username"], "email": acc.get("email", "")}
