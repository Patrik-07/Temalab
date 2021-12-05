import math

from direct.task.TaskManagerGlobal import taskMgr
from panda3d.core import Texture, CardMaker, WindowProperties, DirectionalLight
from direct.showbase.ShowBase import ShowBase
from cv2 import cv2
# import mediapipe as mp
from mediapipe.python.solutions.face_mesh import FaceMesh

from nose import get_dist_between_nose_tip_landmarks

mp_face_mesh = FaceMesh(refine_landmarks = True)

class Vec2:
    def __init__(self, a, b):
        self.a = a
        self.b = b


class Window(ShowBase):
    def __init__(self, cap):
        super().__init__()

        self.nose = 4

        self.r1 = 6
        self.r2 = 168
        self.lp1 = None
        self.lp2 = None

        self.process = False
        self.cap = cap
        _, frame = self.cap.read()
        self.h, self.w, _ = frame.shape
        scale = 1

        self.iter = 0
        self.points = []

        print(self.h)
        print(self.w)
        props = WindowProperties()
        props.setSize(scale * self.w, scale * self.h)
        self.win.requestProperties(props)

        self.texture = Texture()
        self.texture.setup2dTexture(self.w, self.h, Texture.T_unsigned_byte, Texture.FRgb8)

        cm = CardMaker('card')
        self.card = self.render.attachNewNode(cm.generate())
        self.card.setPos(-(self.w / self.h) / 2, 3.5, -0.5)
        self.card.setScale(self.w / self.h, 1, 1)

        self.x_offset = (self.w / self.h) / 2
        self.y_offset = 0.5 + 0.05

        self.model = self.loader.loadModel('sphere.obj')
        t = self.loader.loadTexture('red.png')
        self.model.setP(self.model, 90)
        self.model.setTexture(t, 1)
        self.model.setScale(0.1, 0.1, 0.1)
        self.model.setPos(0, 3.5, 0.25)
        self.model.setBin("fixed", 0)
        self.model.setDepthTest(False)
        self.model.setDepthWrite(False)
        self.model.reparentTo(self.render)

        dl = DirectionalLight('dl')
        dl.setColor((1, 1, 1, 1))
        dlnp = self.render.attachNewNode(dl)
        dlnp.setHpr(0, -20, 0)
        self.render.setLight(dlnp)

        taskMgr.add(self.update_tex, 'video frame update')

        self.accept("s", self.on_s_pressed)
        self.accept("i", self.on_i_pressed)
        self.accept("m", self.on_m_pressed)

        self.angle = 0

    def on_s_pressed(self):
        self.process = not self.process

    def on_i_pressed(self):
        self.iter += 1

    def on_m_pressed(self):
        self.points.append(self.iter)

    def update_tex(self, task):
        success, frame = self.cap.read()
        if success:
            if self.process:
                faces = mp_face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)).multi_face_landmarks
                if faces:
                    face = faces[0]
                landmarks = frame.copy()

                fx = face.landmark[self.nose].x
                fy = face.landmark[self.nose].y

                d = fx - 0.5
                if d != 0:
                    px = self.x_offset / (0.5 / d)
                else:
                    px = 0

                d = 0.5 - fy
                if d != 0:
                    py = self.y_offset / (0.5 / d)
                else:
                    py = 0

                self.model.setX(px)
                self.model.setZ(py)

                nose_scale = get_dist_between_nose_tip_landmarks(frame, face)
                self.model.setScale(
                    nose_scale / frame.shape[0],
                    nose_scale / frame.shape[0],
                    nose_scale / frame.shape[0]
                )

                for i in range(0, 468):
                    x = int(face.landmark[i].x * self.w)
                    y = int(face.landmark[i].y * self.h)
                    cv2.circle(landmarks, (x, y), 2, (53, 200, 243), -1)

                for p in self.points:
                    x = int(face.landmark[p].x * self.w)
                    y = int(face.landmark[p].y * self.h)
                    cv2.circle(landmarks, (x, y), 2, (0, 0, 255), -1)

                x = int(face.landmark[168].x * self.w)
                y = int(face.landmark[168].y * self.h)
                cv2.circle(landmarks, (x, y), 2, (0, 0, 255), -1)
                x = int(face.landmark[self.iter].x * self.w)
                y = int(face.landmark[self.iter].y * self.h)
                cv2.circle(landmarks, (x, y), 2, (255, 0, 0), -1)
                frame = cv2.addWeighted(frame, 0.5, landmarks, 0.5, 0)
                

            img = cv2.flip(frame, 0)
            self.texture.setRamImageAs(img.tobytes(), "BGR")
            self.card.setTexture(self.texture)

        return task.cont

cap = cv2.VideoCapture(0)
window = Window(cap)
window.run()
