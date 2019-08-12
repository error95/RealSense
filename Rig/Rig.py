
from Camera import Camera
from Robot import Robot


class Rig:

    def __init__(self, init_file_camera=None, init_file_robot=None):
        cameras_calibrated = False
        robot_calibrated = False

        self.camera = Camera(init_file_camera)

        self.robot_init_data = self.camera.get_robot_data()

        self.robot = Robot(self.robot_init_data)

