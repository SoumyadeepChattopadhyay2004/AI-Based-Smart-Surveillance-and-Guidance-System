"""
mongo.py
Single MongoDB Atlas connection shared by the whole app: the `db` handle for
structured collections (users, attendance, alerts, ...) and a GridFS bucket
`fs` for binary data (face samples, the trained model, alert snapshots).

Everything the app persists lives in Atlas. Nothing here writes to local
disk — see storage.py for the few spots where OpenCV forces a short-lived
temp file to load/save a model.
"""

from pymongo import MongoClient, ReturnDocument
import gridfs

import config

_client = MongoClient(config.MONGODB_URI)
db = _client[config.MONGODB_DB_NAME]
fs = gridfs.GridFSBucket(db, bucket_name="media")


def next_id(counter_name):
    """
    Emulates SQLite's AUTOINCREMENT with a per-collection counter document,
    so existing integer ids (used throughout app.py / templates, e.g.
    /capture/<int:user_id>) keep working unchanged against Mongo's _id.
    """
    doc = db.counters.find_one_and_update(
        {"_id": counter_name},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return doc["seq"]
