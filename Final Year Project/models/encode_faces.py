import face_recognition
import pickle
import cv2
import os
import sys

# Allow running this file directly (python models/encode_faces.py)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import DATASET_PATH, ENCODINGS_PATH

known_encodings = []
known_names = []

# ✅ Check dataset folder
if not os.path.isdir(DATASET_PATH):
    raise FileNotFoundError(f"Dataset folder not found: {DATASET_PATH}")

print("🔄 Processing dataset...")

# ✅ Loop through each user folder
for person_name in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_path):
        continue

    print(f"👤 Encoding user: {person_name}")

    # Loop through images of that user
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        image = cv2.imread(img_path)

        if image is None:
            print(f"⚠️ Skipping unreadable image: {img_name}")
            continue

        # ✅ 🔥 BLUR CHECK (NEW ADDITION)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 50:
            print(f"⚠️ Blurry image skipped: {img_name}")
            continue

        # Convert to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb)

        if len(boxes) == 0:
            print(f"⚠️ No face found in: {img_name}")
            continue

        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(person_name)

print(f"📊 Total encodings: {len(known_encodings)}")

# ✅ Save encodings
data = {
    "encodings": known_encodings,
    "names": known_names
}

with open(ENCODINGS_PATH, "wb") as f:
    pickle.dump(data, f)

print("✅ Model trained & encodings saved successfully!")