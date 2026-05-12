import os
import pickle

import cv2
import face_recognition

from config import ENCODINGS_PATH


_cached_data = {"encodings": [], "names": []}
_cached_mtime = None


def _load_encodings_from_disk():
    if not os.path.exists(ENCODINGS_PATH):
        return {"encodings": [], "names": []}

    with open(ENCODINGS_PATH, "rb") as f:
        loaded = pickle.load(f)

    if not isinstance(loaded, dict):
        return {"encodings": [], "names": []}

    return {
        "encodings": loaded.get("encodings", []),
        "names": loaded.get("names", []),
    }


def _get_latest_encodings():
    """Reload encodings if models/encodings.pkl has changed."""
    global _cached_data, _cached_mtime

    if not os.path.exists(ENCODINGS_PATH):
        _cached_data = {"encodings": [], "names": []}
        _cached_mtime = None
        return _cached_data

    current_mtime = os.path.getmtime(ENCODINGS_PATH)
    if _cached_mtime != current_mtime:
        _cached_data = _load_encodings_from_disk()
        _cached_mtime = current_mtime

    return _cached_data


def recognize_face(frame, tolerance: float = 0.5):
    """Return matched user name or None for unknown/no-face."""
    result = recognize_face_details(frame, tolerance=tolerance)
    return result["name"]


def recognize_face_details(frame, tolerance: float = 0.5):
    """
    Return structured recognition details for the first detected face.
    keys: name, box, confidence, face_found
    """
    data = _get_latest_encodings()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, number_of_times_to_upsample=1, model="hog")

    if not boxes:
        return {"name": None, "box": None, "confidence": 0.0, "face_found": False}

    if not data["encodings"]:
        return {
            "name": None,
            "box": boxes[0],
            "confidence": 0.0,
            "face_found": True,
        }

    encodings = face_recognition.face_encodings(rgb, boxes)
    for idx, encoding in enumerate(encodings):
        distances = face_recognition.face_distance(data["encodings"], encoding)
        if len(distances) == 0:
            continue

        best_idx = distances.argmin()
        confidence = float(max(0.0, min(1.0, 1.0 - distances[best_idx])))
        if distances[best_idx] <= tolerance:
            return {
                "name": data["names"][best_idx],
                "box": boxes[idx],
                "confidence": confidence,
                "face_found": True,
            }

    return {
        "name": None,
        "box": boxes[0],
        "confidence": 0.0,
        "face_found": True,
    }
