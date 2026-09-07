"""
storage.py
Binary file storage on MongoDB Atlas via GridFS — face sample images, the
trained LBPH model + label map, and alert snapshots. Replaces the old
known_faces/, trainer/ and alert_snapshots/ local folders entirely.

OpenCV's cv2.face.LBPHFaceRecognizer can only load/save via a filesystem
path (no in-memory API), so training and loading briefly write to
tempfile.TemporaryDirectory() / NamedTemporaryFile — always cleaned up
immediately after use, never left on disk.
"""

import os
import tempfile

from mongo import db as _db, fs

_files = _db.media.files  # GridFS's own metadata collection


# ---------------------------------------------------------------------------
# Face sample images  (metadata: kind="face_sample", user_id, index)
# ---------------------------------------------------------------------------
def save_face_sample(user_id, index, image_bytes):
    fs.upload_from_stream(
        f"{user_id}_{index}.jpg",
        image_bytes,
        metadata={"kind": "face_sample", "user_id": user_id, "index": index},
    )


def count_face_samples(user_id):
    return _files.count_documents({"metadata.kind": "face_sample", "metadata.user_id": user_id})


def user_ids_with_samples():
    return sorted(_files.distinct("metadata.user_id", {"metadata.kind": "face_sample"}))


def delete_face_samples(user_id):
    for f in _files.find({"metadata.kind": "face_sample", "metadata.user_id": user_id}):
        fs.delete(f["_id"])


def download_user_samples(user_id, dest_dir):
    """Writes every stored sample for user_id into dest_dir as JPGs, returns their paths."""
    paths = []
    for f in _files.find({"metadata.kind": "face_sample", "metadata.user_id": user_id}):
        path = os.path.join(dest_dir, f["filename"])
        with open(path, "wb") as out:
            fs.download_to_stream(f["_id"], out)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Trained LBPH model (binary, GridFS) + label map (small JSON doc, plain Mongo)
# ---------------------------------------------------------------------------
def save_trainer_model(model_bytes):
    for f in _files.find({"metadata.kind": "trainer_model"}):
        fs.delete(f["_id"])
    fs.upload_from_stream("trainer.yml", model_bytes, metadata={"kind": "trainer_model"})


def load_trainer_model_bytes():
    f = _files.find_one({"metadata.kind": "trainer_model"}, sort=[("uploadDate", -1)])
    if not f:
        return None
    buf = bytearray()
    fs.download_to_stream(f["_id"], _ByteSink(buf))
    return bytes(buf)


class _ByteSink:
    """Minimal file-like object so GridFS can stream straight into a bytearray."""
    def __init__(self, buf):
        self._buf = buf

    def write(self, chunk):
        self._buf.extend(chunk)


def save_labels(label_map):
    _db.trainer_meta.replace_one(
        {"_id": "labels"},
        {"_id": "labels", "map": {str(k): v for k, v in label_map.items()}},
        upsert=True,
    )


def load_labels():
    doc = _db.trainer_meta.find_one({"_id": "labels"})
    if not doc:
        return {}
    return {int(k): v for k, v in doc["map"].items()}


# ---------------------------------------------------------------------------
# Alert / event snapshots  (metadata: kind="alert_snapshot")
# ---------------------------------------------------------------------------
def save_alert_snapshot(image_bytes, filename):
    file_id = fs.upload_from_stream(filename, image_bytes, metadata={"kind": "alert_snapshot"})
    return str(file_id)  # stored as alerts.snapshot_path — a GridFS file id


def load_snapshot_bytes(file_id_str):
    from bson import ObjectId
    buf = bytearray()
    fs.download_to_stream(ObjectId(file_id_str), _ByteSink(buf))
    return bytes(buf)
