import math

from cv2 import cv2
from direct.task.TaskManagerGlobal import taskMgr
from panda3d.core import Texture, CardMaker, WindowProperties, DirectionalLight, Material
from direct.showbase.ShowBase import ShowBase
from enum import IntEnum
from mediapipe.python.solutions import face_mesh

from nose import get_dist_between_nose_tip_landmarks

mp_face_mesh = face_mesh.FaceMesh(refine_landmarks=True)
LANDMARKS_N = face_mesh.FACEMESH_NUM_LANDMARKS_WITH_IRISES

class Obj(IntEnum):
    NOSE = 0
    MOUSTACHE = 1

OBJECT_DATA = {
    Obj.NOSE: {
        "obj_file_path": "res/sphere.obj",
        "texture_file_path": "res/sphere.png",
        "initial_scale": (0.1, 0.1, 0.1),
        "scale_factor": 1,
        "landmark_idx": 4
    }, 
    Obj.MOUSTACHE: {
        "obj_file_path": "res/moustache.obj",
        "texture_file_path": "res/moustache.bmp",
        "initial_scale": (0.1, 0.1, 0.1),
        "scale_factor": 0.01,
        "landmark_idx": 164
    }
}

class Window(ShowBase):
    def __init__(self, cap):
        super().__init__()

        self.init_window()

        self.models = []

        self.loadObjects()
        self.model_object = Obj.NOSE
        self.model = self.models[int(self.model_object)]

        self.init_directional_light()

        taskMgr.add(self.update_tex, 'video frame update')

        self.accept("s", self.on_s_pressed)
        self.accept("i", self.on_i_pressed)
        self.accept("u", self.on_u_pressed)
        self.accept("n", self.on_n_pressed)
        self.accept("m", self.on_m_pressed)

        self.angle = 0

    def init_window(self):
        self.process = False
        self.cap = cap
        _, frame = self.cap.read()
        self.h, self.w, _ = frame.shape
        scale = 1

        self.landmark_idx = 0

        print(self.h)
        print(self.w)
        props = WindowProperties()
        props.setSize(scale * self.w, scale * self.h)
        self.win.requestProperties(props)

        self.texture = Texture()
        self.texture.setup2dTexture(
            self.w, self.h, Texture.T_unsigned_byte, Texture.FRgb8)

        cm = CardMaker('card')
        self.card = self.render.attachNewNode(cm.generate())
        self.card.setPos(-(self.w / self.h) / 2, 3.5, -0.5)
        self.card.setScale(self.w / self.h, 1, 1)

        self.x_offset = (self.w / self.h) / 2
        self.y_offset = 0.5 + 0.05

    def init_directional_light(self):
        dl = DirectionalLight('dl')
        dl.setColor((1, 1, 1, 1))
        dlnp = self.render.attachNewNode(dl)
        dlnp.setHpr(0, -20, 0)
        self.render.setLight(dlnp)

    def loadObjects(self):
        for key in OBJECT_DATA.keys():
            data = OBJECT_DATA[key]

            # getting data
            obj_file_path = data["obj_file_path"]
            texture_file_path = data["texture_file_path"]
            scale_factor = data["scale_factor"]
            scale_x, scale_y, scale_z = tuple(scale_factor * scl for scl in data["initial_scale"])
            
            # loading the model
            model = self.loader.loadModel(obj_file_path)
            
            t = self.loader.loadTexture(texture_file_path)
            model.setTexture(t)
            model.setP(model, 90)

            model.setScale(scale_x, scale_y, scale_z)
            model.setPos(0, 3.5, 0.25)
            model.setBin("fixed", 0)
            model.setDepthTest(False)
            model.setDepthWrite(False)

            # model.reparentTo(self.render)

            # adding model to models array
            self.models.append(model)

    def init_model(self):
        self.models[int(self.model_object)].reparentTo(self.render)
        self.models[int(self.model_object)].setPos(0, 3.5, 0.25)
        self.model = self.models[int(self.model_object)]

    def on_s_pressed(self):
        self.process = not self.process

    def on_i_pressed(self):
        self.landmark_idx = (self.landmark_idx + 1) % LANDMARKS_N

    def on_u_pressed(self):
        self.landmark_idx = (self.landmark_idx - 1) % LANDMARKS_N

    def on_n_pressed(self):
        self.process = False
        self.models[int(self.model_object)].detachNode()
        self.model_object = Obj.NOSE
        self.init_model()

    def on_m_pressed(self):
        self.process = False
        self.models[int(self.model_object)].detachNode()
        self.model_object = Obj.MOUSTACHE
        self.init_model()

    def is_model_nose(self):
        return self.model == self.models[int(Obj.NOSE)]

    def is_model_moustache(self):
        return self.model == self.models[int(Obj.MOUSTACHE)]
    
    def get_model_scale(self, frame, face):
        model_scale = get_dist_between_nose_tip_landmarks(frame, face)
    
        if self.is_model_nose():
            scale_factor = OBJECT_DATA[Obj.NOSE]["scale_factor"]
        if self.is_model_moustache():
            scale_factor = OBJECT_DATA[Obj.MOUSTACHE]["scale_factor"]
        
        return model_scale * scale_factor

    def update_tex(self, task):
        success, frame = self.cap.read()
        if success:
            if self.process:
                faces = mp_face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)).multi_face_landmarks
                face = None
                if faces:
                    face = faces[0]
                landmarks = frame.copy()
                
                if face:
                    self.updateModel(frame, face)

                    # for i in range(0, LANDMARKS_N):
                    #     x = int(face.landmark[i].x * self.w)
                    #     y = int(face.landmark[i].y * self.h)
                    #     cv2.circle(landmarks, (x, y), 2, (53, 200, 243), -1)

                frame = cv2.addWeighted(frame, 0.5, landmarks, 0.5, 0) 

            img = cv2.flip(frame, 0)
            self.texture.setRamImageAs(img.tobytes(), "BGR")
            self.card.setTexture(self.texture)

        return task.cont

    def updateModel(self, frame, face):
        model_scale = self.get_model_scale(frame, face)

        self.model.setScale(
            model_scale / frame.shape[0],
            model_scale / frame.shape[0],
            model_scale / frame.shape[0]
        )

        landmark_idx = OBJECT_DATA[int(self.model_object)]["landmark_idx"]
        fx = face.landmark[landmark_idx].x
        fy = face.landmark[landmark_idx].y

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

cap = cv2.VideoCapture(0)
window = Window(cap)
window.run()
