import face_recognition
import pickle
import os
import cv2
import numpy as np

DATA_FILE = "data/face_encodings.pkl"

def load_face_data():
    """Load existing face data from file."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def save_face_data(face_data):
    """Save face data to file."""
    os.makedirs("data", exist_ok=True)
    with open(DATA_FILE, 'wb') as f:
        pickle.dump(face_data, f)

def recognize_faces(image_np, known_face_data):
    """
    Recognize faces in a given image.
    Returns a list of dictionaries, one for each detected face.
    """
    if not known_face_data:
        return []

    # Prepare known faces data
    known_face_encodings = []
    known_face_names = []
    for name, encodings in known_face_data.items():
        for encoding in encodings:
            known_face_encodings.append(encoding)
            known_face_names.append(name)
            
    # Convert image to RGB
    rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
    # Find all face locations and encodings
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    img_height, img_width, _ = image_np.shape
    detected_faces = []

    for i, face_encoding in enumerate(face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"
        confidence = 0.0

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]

        top, right, bottom, left = face_locations[i]
        box_percent = [
            (top / img_height) * 100,
            (right / img_width) * 100,
            (bottom / img_height) * 100,
            (left / img_width) * 100,
        ]

        detected_faces.append({
            "id": f"{name}-{i}",
            "name": name,
            "confidence": confidence,
            "box": box_percent,
        })

    return detected_faces