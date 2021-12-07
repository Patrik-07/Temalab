import cv2
from mediapipe.python.solutions import face_mesh as mp_face_mesh

LANDMARKS_N = mp_face_mesh.FACEMESH_NUM_LANDMARKS
REFINED_LANDMARKS_N = mp_face_mesh.FACEMESH_NUM_LANDMARKS_WITH_IRISES

image = cv2.imread("patterns/zucc.png")

face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Facial landmarks
result = face_mesh.process(rgb_image)

height, width, _ = image.shape

for facial_landmarks in result.multi_face_landmarks:
    for i in range(0, LANDMARKS_N):
        pt1 = facial_landmarks.landmark[i]
        x = int(pt1.x * width)
        y = int(pt1.y * height)
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
    for i in range(LANDMARKS_N, REFINED_LANDMARKS_N):
        pt1 = facial_landmarks.landmark[i]
        x = int(pt1.x * width)
        y = int(pt1.y * height)
        cv2.circle(image, (x, y), 2, (100, 255, 0), -1)
cv2.imshow("Image", image)
cv2.waitKey(0)
