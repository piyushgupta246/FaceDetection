import os
import cv2
from config import DATASET_PATH


def capture_images(name: str, target_count: int = 20) -> None:
    """Capture sample face images for a user and store them in dataset/user_images/<name>/."""
    safe_name = (name or "").strip()
    if not safe_name:
        raise ValueError("User name is required to capture dataset images")

    user_dir = os.path.join(DATASET_PATH, safe_name)
    os.makedirs(user_dir, exist_ok=True)

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Unable to open camera for dataset capture")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    captured = 0
    frame_step = 0
    while captured < target_count:
        ok, frame = camera.read()
        if not ok:
            continue

        frame_step += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save only if face is detected and every few frames to avoid duplicates
        if len(faces) > 0 and frame_step % 4 == 0:
            file_path = os.path.join(user_dir, f"{captured + 1}.jpg")
            cv2.imwrite(file_path, frame)
            captured += 1

        cv2.putText(
            frame,
            f"Capturing {captured}/{target_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        if len(faces) == 0:
            cv2.putText(
                frame,
                "No face detected - align your face in camera",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Register - Capture Dataset", frame)

        # Allow quitting early via q key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()



