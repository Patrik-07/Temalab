from math import sqrt
from mediapipe.python.solutions import face_mesh, face_mesh_connections

def get_nose_tip_landmarks(face) -> list:
    """
    Args:
        face: contains facial landmark data, that can be accessed though the landmark property,
            which is a list of landmarks
    
    Returns:
        A two element list of facial landmarks. the first is the lower tip of the nose, the second is the upper tip.
    """
    return [face.landmark[19], face.landmark[195]]

def get_dist_between_nose_tip_landmarks(image, face) -> float:
    landmarks = get_nose_tip_landmarks(face)
    first_tip, second_tip = landmarks[0], landmarks[1]

    height, width = image.shape[0], image.shape[1]
    
    x1, y1 = int(first_tip.x * width), int(first_tip.y * height)
    x2, y2 = int(second_tip.x * width), int(second_tip.y * height)
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
