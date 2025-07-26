import base64
import numpy as np
import cv2

def base64_to_image(base64_string):
    """
    Converts a base64 encoded string to a numpy array (OpenCV image).
    """
    if "," in base64_string:
        base64_string = base64_string.split(',')[1]
    
    img_data = base64.b64decode(base64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    return image