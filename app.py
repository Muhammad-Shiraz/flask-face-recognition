import cv2
import face_recognition
import pickle
import numpy as np
import threading
import requests  # <-- Import requests to send data to Django
import json
from flask import Flask, Response, request, jsonify, render_template
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# Load face encodings
with open("employee_faces.pkl", "rb") as file:
    face_data = pickle.load(file)

print(f"✅ Loaded {len(face_data)} employee encodings.")

# Global variables
camera = None
is_running = False
frame_lock = threading.Lock()
latest_frame = None

DJANGO_API_URL =  "https://your-django-app.up.railway.app/attendance/mark/"
  # Django endpoint

# Function to send attendance to Django
def send_attendance(name):
    """Send recognized employee name to Django for attendance marking."""
    try:
        response = requests.post(DJANGO_API_URL, json={"name": name})
        print("✅ Django Response:", response.json())
    except Exception as e:
        print("❌ Error sending attendance:", e)


def load_encodings():
    """ Load updated face encodings from the pickle file. """
    global face_data
    try:
        with open("employee_faces.pkl", "rb") as file:
            face_data = pickle.load(file)
        print(f"✅ Reloaded {len(face_data)} employee encodings.")
    except Exception as e:
        print(f"❌ Error loading encodings: {e}")







# Face recognition processing

def process_camera():
    global camera, is_running, latest_frame
    camera = cv2.VideoCapture(0)

    while is_running:
        success, frame = camera.read()
        if not success:
            continue

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Face detection
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        print(f"Detected {len(face_locations)} face(s) in the frame")

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(list(face_data.values()), face_encoding, tolerance=0.5)
            distances = face_recognition.face_distance(list(face_data.values()), face_encoding)
            print(f"Face match distances: {distances}")

            name = "Unknown"
            attendance_marked = False  # Flag to track if attendance is marked

            if True in matches:
                best_match_index = np.argmin(distances)
                if distances[best_match_index] < 0.4:  # Adjust if needed
                    name = list(face_data.keys())[best_match_index]
                    print(name)

                    # ✅ Call Django API to mark attendance
                    response = requests.post("http://127.0.0.1:8000/attendance/mark/", json={"employee_id": name})
                    if response.status_code == 200:
                        attendance_marked = True  # Mark attendance successfully
                        print(f"✅ Attendance marked for {name}")

            # Draw Rectangle & Name
            top, right, bottom, left = [v * 4 for v in face_location]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            # ✅ Display "Attendance Marked" message if attendance was recorded
            if attendance_marked:
                cv2.putText(frame, "Attendance Marked", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        with frame_lock:
            latest_frame = frame.copy()

    camera.release()

@app.route('/')
def face_page():
    return render_template("face.html")  # ✅ Load the face.html page








@app.route('/camera_status', methods=['GET'])
def camera_status():
    global is_running
    return jsonify({"running": is_running}), 200
# API to start camera
@app.route("/start_camera", methods=["POST"])
def start_camera():
    global is_running
    if not is_running:
        is_running = True
        threading.Thread(target=process_camera, daemon=True).start()
        return jsonify({"message": "Camera started"}), 200
    return jsonify({"message": "Camera already running"}), 200

# API to stop camera
@app.route("/stop_camera", methods=["POST"])
def stop_camera():
    global is_running, camera
    is_running = False

    if camera:
        camera.release()  # ✅ Release the camera properly
        camera = None  # ✅ Reset the camera object

    return jsonify({"message": "Camera stopped"}), 200


# Video Stream
def generate_feed():
    global latest_frame
    while is_running:
        with frame_lock:
            if latest_frame is None:
                continue
            _, jpeg = cv2.imencode(".jpg", latest_frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(generate_feed(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/reload_encodings", methods=["POST"])
def reload_encodings():
    global face_data
    try:
        with open("employee_faces.pkl", "rb") as file:
            face_data = pickle.load(file)  # ✅ Reload encodings in memory
        print("✅ Flask: Face encodings reloaded successfully!")
        return jsonify({"message": "Encodings reloaded"}), 200
    except Exception as e:
        print(f"❌ Flask: Error reloading encodings: {e}")
        return jsonify({"error": "Failed to reload encodings"}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
