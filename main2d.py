from cv2 import cv2
import mediapipe as mp
import numpy as np
import cv2

# for screen capture
from mss import mss
from PIL import Image

import mouth
import iris

# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

green = (0, 255, 0)
blue = (255, 0, 0)
red = (0, 0, 255)
white = (255, 255, 255)
black = (0, 0, 0)
firebrick = (139, 0, 0)

class Process(Enum):
    NOTHING = 1
    MOUTH = 2
    GLASSES = 3

"""
def mouse_move(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        img = frame.copy()
        a = mouth.get_mouth_landmarks(faces[0])
        for a in mouth_landmarks:
            pdx = int(mouth_landmark.x * width)
            pdy = int(mouth_landmark.y * height)
            cv2.circle(frame, (pdx, pdy), 2, dot_color1, -1)
            if abs(pdx - x) < 5 and abs(pdy - y) < 5:
                cv2.circle(frame, (pdx, pdy), 2, dot_color1, -1)
                print(i)
            cv2.imshow('FaceMesh', img)
"""

def mouth_landmarks(image, face):
    """
    Args:
        image: np.ndarray in the tuple returned by cv2.VideoCapture.read()
        face: contains facial landmark data, that can be accessed though the landmark property,
            which is a list of landmarks

    Returns:
        An ndarray of the image with the mouth landmarks drawn as green circles
    """
    img = image.copy()

    height = img.shape[0]
    width = img.shape[1]

    landmarks = mouth.get_mouth_landmarks(face)
    for point in landmarks:
        x = int(point.x * width)
        y = int(point.y * height)
        cv2.circle(img, (x, y), 1, green, -1)

    return img


def iris_mask(image, face):
    """
    Args:
        image: np.ndarray in the tuple returned by cv2.VideoCapture.read()
        face: contains facial landmark data, that can be accessed though the landmark property,
            which is a list of landmarks

    Returns:
        An ndarray representing the iris mask image (black everywhere except the iris, which is
        a white polygon)
    """
    img = image.copy()

    height = img.shape[0]
    width = img.shape[1]

    left_iris = iris.get_left_iris_landmarks(face)
    right_iris = iris.get_right_iris_landmarks(face)

    mask = np.zeros_like(img)

    mask_points = []
    for point in left_iris:
        x = int(point.x * width)
        y = int(point.y * height)
        mask_points.append([x, y])
    mask = cv2.fillPoly(mask, [np.array(mask_points)], white)

    mask_points = []
    for point in right_iris:
        x = int(point.x * width)
        y = int(point.y * height)
        mask_points.append([x, y])
    mask = cv2.fillPoly(mask, [np.array(mask_points)], white)

    return mask


def mouth_mask(image, face):
    """
    Args:
        image: np.ndarray in the tuple returned by cv2.VideoCapture.read()
        face: contains facial landmark data, that can be accessed though the landmark property,
            which is a list of landmarks
    
    Returns:
        An ndarray representing the mouth mask image (black everywhere except the lips, where 
        it is white)
    """
    img = image.copy()

    height = img.shape[0]
    width = img.shape[1]

    inner_mouth = mouth.get_inner_mouth(face)
    outer_mouth = mouth.get_outer_mouth(face)

    mask = np.zeros_like(img)

    mask_points = []
    for point in outer_mouth:
        x = int(point.x * width)
        y = int(point.y * height)
        mask_points.append([x, y])
    mask = cv2.fillPoly(mask, [np.array(mask_points)], white)

    mask_points = []
    for point in inner_mouth:
        x = int(point.x * width)
        y = int(point.y * height)
        mask_points.append([x, y])
    mask = cv2.fillPoly(mask, [np.array(mask_points)], black)

    return mask


def mouth_bounds(image, face):
    """
    Args:
        image: np.ndarray in the tuple returned by cv2.VideoCapture.read()
        face: contains facial landmark data, that can be accessed though the landmark property,
            which is a list of landmarks

    Returns:
        An ndarray representing the mouth bounds image (the input image with a blue contour
        around the outside of the mouth and a red contour around the inside)
    """
    img = image.copy()

    height = img.shape[0]
    width = img.shape[1]

    inner_mouth = mouth.get_inner_mouth(face)
    outer_mouth = mouth.get_outer_mouth(face)

    line_points = []
    for point in outer_mouth:
        x = int(point.x * width)
        y = int(point.y * height)
        line_points.append([x, y])
        cv2.polylines(img, [np.array(line_points)], False, blue, 2)
    cv2.line(img, line_points[0], line_points[len(line_points) - 1], blue, 2)

    line_points = []
    for point in inner_mouth:
        x = int(point.x * width)
        y = int(point.y * height)
        line_points.append([x, y])
        cv2.polylines(img, [np.array(line_points)], False, red, 2)
    cv2.line(img, line_points[0], line_points[len(line_points) - 1], red, 2)

    return img


def iris_color(image, face, color):
    """
    Args:
        image: np.ndarray in the tuple returned by cv2.VideoCapture.read()
        face: contains facial landmark data, that can be accessed though the landmark property,
            which is a list of landmarks
        color: list of integers between 0 and 255, representing rgba data: 
            [0,0,0,0] <=> rgba(0,0,0,0)

    Returns:
        An ndarray representing the iris color image (the input image with colored polygons on 
        the irises)
    """
    img = image.copy()

    mask = iris_mask(img, face)

    colored_iris = np.zeros_like(mask)
    colored_iris[:] = color[:3]
    colored_iris = cv2.bitwise_and(mask, colored_iris)
    colored_iris = cv2.GaussianBlur(colored_iris, (15, 15), cv2.BORDER_DEFAULT)
    colored_iris = cv2.addWeighted(img, 1, colored_iris, color[3] / 256, 0)

    return colored_iris


def mouth_color(image, face, color):
    """
    Args:
        image: np.ndarray in the tuple returned by cv2.VideoCapture.read()
        face: contains facial landmark data, that can be accessed though the landmark property,
            which is a list of landmarks
        color: list of integers between 0 and 255, representing rgba data: 
            [0,0,0,0] <=> rgba(0,0,0,0)

    Returns:
        An ndarray representing the mouth color image (the input image with the mouth colored)
    """
    img = image.copy()

    mask = mouth_mask(img, face)

    colored_mouth = np.zeros_like(mask)
    colored_mouth[:] = color[:3]
    colored_mouth = cv2.bitwise_and(mask, colored_mouth)
    colored_mouth = cv2.GaussianBlur(colored_mouth, (5, 5), cv2.BORDER_DEFAULT)
    colored_mouth = cv2.addWeighted(img, 1, colored_mouth, color[3] / 256, 0)

    return colored_mouth


def mouth_pattern(image, face, pattern_path):
    img = image.copy()

    mask = mouth_mask(img, face)
    pattern = cv2.imread(pattern_path)

    pattern = cv2.resize(pattern, (mask.shape[1], mask.shape[0]))
    pattern_mouth = cv2.bitwise_and(pattern, mask)
    pattern_mouth = cv2.GaussianBlur(pattern_mouth, (3, 3), cv2.BORDER_DEFAULT)
    pattern_mouth = cv2.addWeighted(img, 1, pattern_mouth, 0.6, 0)

    return pattern_mouth


def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])


def nothing(x):
    pass


def create_trackbars():
    cv2.namedWindow('Color')
    cv2.resizeWindow('Color', 500, 400)

    cv2.createTrackbar('Mouth R', 'Color', 0, 255, nothing)
    cv2.createTrackbar('Mouth G', 'Color', 0, 255, nothing)
    cv2.createTrackbar('Mouth B', 'Color', 0, 255, nothing)
    cv2.createTrackbar('Mouth A', 'Color', 0, 255, nothing)

    cv2.createTrackbar('Iris R', 'Color', 0, 255, nothing)
    cv2.createTrackbar('Iris G', 'Color', 0, 255, nothing)
    cv2.createTrackbar('Iris B', 'Color', 0, 255, nothing)
    cv2.createTrackbar('Iris A', 'Color', 0, 255, nothing)


def get_mouth_color_from_trackbar():
    r = cv2.getTrackbarPos('Mouth R', 'Color')
    g = cv2.getTrackbarPos('Mouth G', 'Color')
    b = cv2.getTrackbarPos('Mouth B', 'Color')
    a = cv2.getTrackbarPos('Mouth A', 'Color')

    return [b, g, r, a]

def get_iris_color_from_trackbar():
    r = cv2.getTrackbarPos('Iris R', 'Color')
    g = cv2.getTrackbarPos('Iris G', 'Color')
    b = cv2.getTrackbarPos('Iris B', 'Color')
    a = cv2.getTrackbarPos('Iris A', 'Color')

    return [b, g, r, a]


def screen_input():
    process = False
    with mss() as sct:
        while True:
            screen_shot = sct.grab(sct.monitors[2])
            frame = Image.frombytes('RGB', (screen_shot.width, screen_shot.height), screen_shot.rgb)
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (854, 360))
            frame = frame[0:0 + 360, 100:100 + 640]

            image = frame

            if process:
                faces = mp_face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)).multi_face_landmarks
                face = faces[0]

                # irises = mp_face_mesh.FACEMASH_IRISES

                color = get_mouth_color_from_trackbar()

                landmarks = mouth_landmarks(frame, face)
                bounds = mouth_bounds(frame, face)
                mask = mouth_mask(frame, face)
                pattern_mouth = mouth_pattern(frame, face, 'patterns/pattern03.jpg')
                colored_mouth = mouth_color(frame, face, color)

                image = concat_tile([[frame, landmarks, bounds], [mask, pattern_mouth, colored_mouth]])

            cv2.imshow('MouthDetect', image)

            if cv2.waitKey(33) & 0xFF in (ord('q'), 27):
                break

            if cv2.waitKey(33) & 0xFF in (ord('s'), 27):
                process = not process
                if process:
                    create_trackbars()
                else:
                    cv2.destroyWindow('MouthColor')

    cv2.destroyAllWindows()

def webcam_input():
    process = Process.NOTHING
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        image = frame

        if process == Process.MOUTH:
            faces = mp_face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)).multi_face_landmarks

            if faces:
                face = faces[0]

            color = get_mouth_color_from_trackbar()
            color_iris = get_iris_color_from_trackbar()

            landmarks = mouth_landmarks(frame, face)
            bounds = mouth_bounds(frame, face)
            mask = mouth_mask(frame, face)
            pattern_mouth = mouth_pattern(frame, face, 'patterns/pattern01.jpg')
            colored_mouth = mouth_color(frame, face, color)
            colored_iris = iris_color(frame, face, color_iris)
            mask_iris = iris_mask(frame, face)

            image = concat_tile([
                        [frame, landmarks, mask, mask_iris],
                        [bounds, pattern_mouth, colored_mouth, colored_iris]
                    ])
        
        if process == Process.GLASSES:
            pass

        cv2.imshow('MouthDetect', image)

        if cv2.waitKey(33) & 0xFF in (ord('q'), 27):
            break

        if cv2.waitKey(33) & 0xFF in (ord('s'), 27):
            process = Process.MOUTH
            if process:
                create_trackbars()
            else:
                cv2.destroyWindow('MouthColor')

        if cv2.waitKey(33) & 0xFF in (ord('o'), 27):
            process = Process.GLASSES

    cv2.destroyAllWindows()


def main():
    # screen_input()
    webcam_input()

if __name__ == "__main__":
    main()
