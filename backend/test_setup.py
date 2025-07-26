import cv2
import face_recognition
import numpy as np
import os

def test_camera():
    """Test camera functionality"""
    print("Testing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Camera not accessible")
        return False
    
    ret, frame = cap.read()
    if ret:
        print("✅ Camera working correctly")
        cv2.imshow("Camera Test - Press 'q' to quit", frame)
        cv2.waitKey(2000)  # Show for 2 seconds
        cv2.destroyAllWindows()
    
    cap.release()
    return ret

def test_face_recognition():
    """Test face_recognition library"""
    print("Testing face_recognition library...")
    try:
        # Create a dummy image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        face_locations = face_recognition.face_locations(test_image)
        print("✅ face_recognition library working")
        return True
    except Exception as e:
        print(f"❌ face_recognition error: {e}")
        return False

def test_directories():
    """Create necessary directories"""
    print("Creating project directories...")
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/photos", exist_ok=True)
    print("✅ Directories created")

if __name__ == "__main__":
    print("=== Face Recognition System Setup Test ===")
    
    test_directories()
    camera_ok = test_camera()
    face_rec_ok = test_face_recognition()
    
    if camera_ok and face_rec_ok:
        print("\n🎉 All tests passed! Ready to proceed to Phase 2.")
    else:
        print("\n❌ Some tests failed. Please fix the issues before proceeding.")