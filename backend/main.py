from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import cv2
import base64
import time
import threading
import face_recognition as fr

from utils import base64_to_image
from recognition import load_face_data, save_face_data, recognize_faces

# --- CONFIGURATION ---
# Set the video source for the recognition thread.
# - To use the RTSP stream, set VIDEO_SOURCE = RTSP_URL
# - To use the default device webcam, set VIDEO_SOURCE = 0
RTSP_URL = 'rtsp://admin:qwerty123@192.168.2.33:554/Streaming/channels/301'
# VIDEO_SOURCE = 0 # <--- USE THIS FOR DEVELOPM`ENT WITH WEBCAM
VIDEO_SOURCE = RTSP_URL  # <--- UNCOMMENT THIS FOR LIVE CCTV


# --- APP INITIALIZATION ---
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables
thread = None
thread_lock = threading.Lock()
known_face_cache = {}

# --- BACKGROUND THREAD FOR VIDEO PROCESSING ---
def recognition_thread():
    """
    Background thread that connects to the video source,
    performs recognition, and emits results via WebSocket.
    """
    global known_face_cache
    print(f"Starting recognition thread with source: {VIDEO_SOURCE}")
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source at {VIDEO_SOURCE}")
        socketio.emit('stream_error', {'message': f'Failed to connect to camera source: {VIDEO_SOURCE}'})
        return

    while True:
        socketio.sleep(0.05)
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or failed. Reconnecting...")
            cap.release()
            time.sleep(5)
            cap = cv2.VideoCapture(VIDEO_SOURCE)
            continue
        
        # --- Perform Recognition ---
        detections = recognize_faces(frame, known_face_cache)

        # --- Encode Frame for Transmission ---
        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        # --- Emit Data to Frontend ---
        socketio.emit('video_frame', {
            'image': 'data:image/jpeg;base64,' + frame_b64,
            'detections': detections
        })

    cap.release()
    print("Recognition thread stopped.")

# --- SOCKET.IO EVENT HANDLERS ---
@socketio.on('connect')
def handle_connect():
    global thread
    with thread_lock:
        if thread is None:
            global known_face_cache
            known_face_cache = load_face_data()
            print(f"Loaded {len(known_face_cache)} known faces.")
            thread = socketio.start_background_task(target=recognition_thread)
    print('Client connected')

@socketio.on('reload_faces')
def handle_reload_faces():
    global known_face_cache
    known_face_cache = load_face_data()
    print(f"Face data reloaded. Tracking {len(known_face_cache)} faces.")
    emit('faces_reloaded', {'count': len(known_face_cache)})

# --- STANDARD HTTP ROUTES ---
@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_stats():
    # This remains the same
    face_data = load_face_data()
    total_people = len(face_data)
    total_photos = sum(len(encodings) for encodings in face_data.values())
    return jsonify({"totalPeople": total_people, "totalPhotos": total_photos})

@app.route('/api/people', methods=['GET'])
def get_registered_people():
    # This remains the same
    face_data = load_face_data()
    people = [{"id": i, "name": name, "photos": len(encodings)} for i, (name, encodings) in enumerate(face_data.items())]
    return jsonify(people)

@app.route('/api/register', methods=['POST'])
def register_person():
    # The registration logic is still initiated from the frontend webcam
    data = request.get_json()
    name = data.get('name')
    photos_b64 = data.get('photos')
    if not name or not photos_b64:
        return jsonify({"error": "Name and photos are required"}), 400

    face_encodings = []
    for photo_b64 in photos_b64:
        image = base64_to_image(photo_b64['src'])
        encodings = fr.face_encodings(image)
        if encodings:
            face_encodings.append(encodings[0])

    if not face_encodings:
        return jsonify({"error": "No faces could be detected"}), 400

    all_face_data = load_face_data()
    all_face_data[name] = face_encodings
    save_face_data(all_face_data)
    
    socketio.emit('reload_faces_event')
    return jsonify({"message": f"Successfully registered {name}"}), 201

if __name__ == '__main__':
    os.makedirs("data", exist_ok=True)
    print("Starting Flask-SocketIO server...")
    socketio.run(app, host='0.0.0.0', port=5001, debug=True, use_reloader=False)