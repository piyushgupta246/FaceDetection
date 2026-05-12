import cv2
import face_recognition
import pickle
import os
from config import ENCODINGS_PATH, DATASET_PATH

def register_new_user(name):
    cap = cv2.VideoCapture(0)

    print("📸 Capturing face... Press 's' to save")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Register Face", frame)

        key = cv2.waitKey(1)

        if key == ord('s'):  # press 's' to capture
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            boxes = face_recognition.face_locations(rgb)

            if len(boxes) == 0:
                print("❌ No face detected")
                continue

            encodings = face_recognition.face_encodings(rgb, boxes)

            # Save image
            img_path = os.path.join(DATASET_PATH, f"{name}.jpg")
            cv2.imwrite(img_path, frame)

            # Load existing encodings
            if os.path.exists(ENCODINGS_PATH):
                with open(ENCODINGS_PATH, "rb") as f:
                    data = pickle.load(f)
            else:
                data = {"encodings": [], "names": []}

            # Add new encoding
            data["encodings"].append(encodings[0])
            data["names"].append(name)

            # Save updated encodings
            with open(ENCODINGS_PATH, "wb") as f:
                pickle.dump(data, f)

            print(f"✅ User '{name}' registered successfully")

            break

    cap.release()
    cv2.destroyAllWindows()