import cv2
from utils.recognition import recognize_face_details

auth_status = "waiting"
auth_meta = {"face_found": False, "confidence": 0.0}


def get_auth_status():
    return auth_status


def get_auth_meta():
    return auth_meta


def reset_auth_status():
    global auth_status, auth_meta
    auth_status = "waiting"
    auth_meta = {"face_found": False, "confidence": 0.0}

def generate_frames():
    global auth_status, auth_meta
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not video.isOpened():
        auth_status = "camera_error"
        return

    # Increase capture reliability on some Windows webcams
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    consecutive_matches = 0
    required_matches = 5

    while True:
        success, frame = video.read()
        if not success:
            auth_status = "camera_error"
            break

        result = recognize_face_details(frame, tolerance=0.52)
        name = result["name"]
        auth_meta = {
            "face_found": result["face_found"],
            "confidence": round(float(result["confidence"]) * 100, 1),
        }

        if result["box"]:
            top, right, bottom, left = result["box"]
            color = (0, 200, 0) if name else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # ✅ If face matched → stop stream
        if name:
            consecutive_matches += 1
            cv2.putText(frame, f"Recognized: {name}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 200, 0), 2)
            if consecutive_matches >= required_matches:
                print(f"✅ Welcome {name}")
                auth_status = "match"
                break
        else:
            consecutive_matches = 0
            auth_status = "waiting" if result["face_found"] else "no_face"

        # Status label for live debugging/clarity
        if result["face_found"] and not name:
            label = "Face found, trying to match..."
            color = (0, 165, 255)
        elif not result["face_found"]:
            label = "No face detected"
            color = (0, 0, 255)
        else:
            label = "Authenticating..."
            color = (255, 255, 255)

        cv2.putText(frame, label, (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if result["face_found"]:
            cv2.putText(frame, f"Confidence: {auth_meta['confidence']}%", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Convert frame for Flask streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    video.release()