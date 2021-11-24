import cv2
import mediapipe as mp

image = cv2.imread("patterns/person.jpg")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Facial landmarks
result = face_mesh.process(rgb_image)

height, width, _ = image.shape

for facial_landmarks in result.multi_face_landmarks:
    for i in range(0, 468):
        pt1 = facial_landmarks.landmark[i]
        x = int(pt1.x * width)
        y = int(pt1.y * height)
        cv2.circle(image, (x, y), 5, (100, 100, 0), -1)

cv2.imshow("Image", image)
cv2.waitKey(0)
