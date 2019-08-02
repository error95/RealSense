import numpy as np
from cv2 import aruco
from stitcher_functions import get_transformation_matrix_wout_rsobject
from stitcher_functions import get_transformation_matrix_wout_rsobject, get_charuco_points, get_charuco_points_ID
from calibration_kabsch_charuco import Transformation


class CameraRobotCalibration:

    def __init__(self, transformation_devices, device_manager):
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        charuco_width = 3
        charuco_height = 3
        square_length = 0.03425
        marker_length = .026

        self.charuco_board = aruco.CharucoBoard_create(charuco_width, charuco_height, square_length, marker_length,
                                                  aruco_dict)
        self.transformation_devices = transformation_devices
        self.device_manager = device_manager
        self.target_to_robot = -1



    def get_robot_to_target(self):
        # Figurde out starting position
        all_corneers_found = False
        start_corners = {}
        for corner in range(len(self.charuco_board.nearestMarkerCorners)):
            start_corners[corner] = np.asarray([0, 0, 0])
        while not all_corneers_found:
            frames = self.device_manager.poll_frames()
            for camera in self.device_manager.devices():
                start_target_points, start_target_IDs = get_charuco_points_ID(frames[camera],
                                                                      self.transformation_devices[camera],
                                                                      self.intrinsics_devices[camera],
                                                                      self.charuco_board)
                for id in start_target_IDs:
                    start_corners[id].append(start_target_points)




        # Move robot

        # Get finishing position
        all_corneers_found = False
        while not all_corneers_found:
            frames = self.device_manager.poll_frames()
            for camera in self.device_manager.devices():
                goal_target_points, goal_target_IDs = get_charuco_points_ID(frames[camera],
                                                                  self.transformation_devices[camera],
                                                                  self.intrinsics_devices[camera],
                                                                  self.charuco_board)
        # Calculate movement
        # Get transfer functions

        # Get robot arm to target
        #Tarbet_tarnsition = Robot_arm_to_target(target_transition, robot_transition)


        # Return target_transition

    def get_camera_to__robot():
        # If robot to target not defined:
            # Get robot to target transition

        # Get camera to target transition

        # return target_to_robot*camera_to_target


    def robot_arm_to_target(AA, BB):
        [m,n] = AA.shape
        n = n/4

        A = np.zeros(9*n, 9)
        b = np.zeros(9*n,1)
        for i in range(1, n+1):
            Ra = AA[0:3, 4*(i-4):4*i-1]
            Rb = BB[0:3, 4*(i-4):4*i-1]
            A[9*i - 9: 9*i,:] = np.kron(Ra, np.eye(3)) + np.kron(-np.eye(3), Rb.T)

        [u, s, v] = np.linalg.svd(A, full_matrices=True)
        x = v[:, -1]
        R = np.reshape(x[0:9], (3, 3)).T
        R = np.sign(np.det(R)/np.abs(np.det(R)))^(1/3)*R
        [u, s, v] = np.linalg.svd(R, full_matrices=True)
        R = u*v.T
        if np.det(R) < 0:
            R = u*np.diag([1, 1, -1])*v.T
        C = np.zeros(3*n, 3)
        d = np.zeros(3*n, 1)
        I = np.eye(3)
        for i in range(1, n+1):
            C[3*i-3: 3*i, :] = I - AA[0:3, 4*i-4:4*i-1]
            d[3*i-3: 3*i, :] = AA[0:3, 4*i] - np.matlmul(R*BB[0:3, 4*i])
        t = C/d
        X = np.eye(4)
        X[0:3, 0:3] = R
        X[3, 0:3] = t

        return X
