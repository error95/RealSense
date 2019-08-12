import numpy as np
from cv2 import aruco
from stitcher_functions import get_transformation_matrix_wout_rsobject
from stitcher_functions import get_transformation_matrix_wout_rsobject, get_charuco_points, get_charuco_points_ID
from calibration_kabsch_charuco import Transformation
from calibration_kabsch_charuco import calculate_transformation_kabsch
import time
from math import cos, sin, radians
import random as rnd


class CameraRobotCalibration:

    def __init__(self, transformation_devices, device_manager, charuco_target, charuco_robot):

        self.charuco_board_target = charuco_target
        self.charuco_board_robot = charuco_robot
        self.transformation_devices = transformation_devices
        self.device_manager = device_manager
        self.X = np.zeros(4, 4)
        self.Y = np.zeros(4, 4)

    def get_charuco_points(self, charuco_board):
        n_corners = len(charuco_board.nearestMarkerCorners)
        found_corners = [False] * n_corners
        start_corners = {}
        average_corners = np.zeros((3, n_corners))
        for corner in range(n_corners):
            start_corners[corner] = np.asarray([[0]]*3)
        while not all(found_corners):
            frames = self.device_manager.poll_frames_keep()
            #found_corners = [False] * n_corners
            for camera in self.device_manager._available_devices:

                start_target_points, start_target_IDs = get_charuco_points_ID(frames[camera],
                                                                              self.transformation_devices[camera],
                                                                              self.device_manager.get_device_intrinsics(frames)[camera],
                                                                              charuco_board)
                for idx, id in enumerate(start_target_IDs):
                    if id:

                        average_corners[:, idx] = start_target_points[:, idx]
                        found_corners[idx] = True

        return average_corners

    def get_target_to_target(self):
        # Figurde out starting position
        input("Set target in staring position")
        start_points_target = self.get_charuco_points(self.charuco_board_target)
        start_points_robot = self.get_charuco_points(self.charuco_board_robot)
        images = 2

        target_transform = np.zeros((4,4*images))
        robot_transform = np.zeros((4,4*images))

        for image in range(images):
            input("Press any key when Target is at a  new position\n")


            # Get finishing position
            end_points_target = self.get_charuco_points(self.charuco_board_target)
            end_points_robot = self.get_charuco_points(self.charuco_board_robot)

            # Calculate movement
            target_movement = calculate_transformation_kabsch(start_points_target, end_points_target)
            transform_matrix = np.hstack((target_movement[0], target_movement[1][:, np.newaxis]))
            print(np.linalg.det(target_movement[0]))
            target_transform[:, image*4: (image+1)*4] = np.vstack((transform_matrix, np.asarray([0, 0, 0, 1])))

            robot_movement = calculate_transformation_kabsch(start_points_robot, end_points_robot)
            transform_matrix = np.hstack((robot_movement[0], robot_movement[1][:, np.newaxis]))
            print(np.linalg.det(robot_movement[0]))
            robot_transform[:, image*4: (image+1)*4] = np.vstack((transform_matrix, np.asarray([0, 0, 0, 1])))

            start_points_robot = end_points_robot
            start_points_target = end_points_target

        coupled = self.robot_arm_to_target_coupled(target_transform, robot_transform)
        decoupled = self.robot_arm_to_target_decoupled(target_transform, robot_transform)

        return self.robot_arm_to_target_coupled(target_transform, robot_transform)

    def robot_arm_to_target_decoupled(self, AA, BB):
        [m,n] = AA.shape
        n = int(n/4)

        A = np.zeros((9*n, 9))
        b = np.zeros((9*n, 1))
        for i in range(1, n+1):
            Ra = AA[0:3, 4*i-4:4*i-1]
            Rb = BB[0:3, 4*i-4:4*i-1]
            A[9*i-9: 9*i, :] = np.kron(Ra, np.eye(3)) + np.kron(-np.eye(3), Rb.T)

        [u, s, v] = np.linalg.svd(A, full_matrices=True)
        v = v.T
        x = v[:, -1]
        R = np.reshape(x[0:9], (3, 3))
        R = (np.sign(np.linalg.det(R))/np.power(np.abs(np.linalg.det(R)), (1.0/3.0)))*R
        [u, s, v] = np.linalg.svd(R, full_matrices=True)
        R = np.matmul(u, v)
        if np.linalg.det(R) < 0:
            R = np.matmul(np.matmul(u, np.diag([1, 1, -1])), v)
        C = np.zeros((3*n, 3))
        d = np.zeros((3*n, 1))
        I = np.eye(3)
        for i in range(1, n+1):
            C[3*i-3: 3*i, :] = I - AA[0:3, 4*i-4:4*i-1]
            temp = AA[0:3, 4*i-1] - np.matmul(R, BB[0:3, 4*i-1])
            d[3*i-3: 3*i, :] = temp[:,np.newaxis]
        t = np.linalg.lstsq(C, d, rcond=None)
        X = np.eye(4)
        X[0:3, 0:3] = R
        X[0:3, 3] = t[0][:,0]

        self.X = X

        return X

    def robot_arm_to_target_coupled(self, AA, BB):
        [m, n] = AA.shape
        n = int(n / 4)

        A = np.zeros((12*n, 12))
        b = np.zeros((12*n,1))
        for i in range(n):
            Ra = AA[0:3, 4*i: 4*i + 3]
            Rb = BB[0:3, 4*i: 4*i + 3]
            ta = AA[0:3, 4*(i+1) - 1]
            tb = BB[0:3, 4*(i+1) - 1]
            A[12*i: 12*i + 9, 0:9] = np.eye(9) - np.kron(Rb, Ra)
            A[12*i + 9: 12*(i+1),:] = np.hstack((np.kron(tb.T, np.eye(3)), np.eye(3) - Ra))
            b[12*i + 9: 12*(i+1)] = ta[:, np.newaxis]
        x = np.linalg.lstsq(A, b, rcond=None)
        X = np.reshape(x[0][0:9], (3, 3))
        X = (np.sign(np.linalg.det(X)) / np.power(np.abs(np.linalg.det(X)), (1.0 / 3.0))) * X
        [u, s, v] = np.linalg.svd(X, full_matrices=False)
        X = np.matmul(u,  v)
        if np.linalg.det(X) < 0:
            X = np.matmul(np.matmul(u,np.diag([1, 1, -1])), v)
        X = np.hstack((X, x[0][9:]))
        X = np.vstack((X, np.asarray([0, 0, 0, 1])))

        self.X = X

        return X

    def robot_arm_target_world_decoupled(self, AA, BB):
        [m, n] = AA.shape
        n = int(n / 4)
        A = np.zeros((9*n, 18))
        T = np.zeros((9, 9))
        b = np.zeros((9*n, 1))

        for i in range(n):
            Ra = AA[0:3, 4*i: 4*i + 3]
            Rb = BB[0:3, 4*i: 4*i + 3]
            T = T + np.kron(Rb, Ra)

        [u, s, v] = np.linalg.svd(T, full_matrices=False)
        v = v.T
        x = v[:, 0]
        y = u[:, 0]

        X = np.reshape(x[0:9], (3, 3))
        X = (np.sign(np.linalg.det(X)) / np.power(np.abs(np.linalg.det(X)), (1.0 / 3.0))) * X
        [u, s, v] = np.linalg.svd(X, full_matrices=False)
        X = np.matmul(u, v)

        Y = np.reshape(y[0:9], (3, 3))
        X = (np.sign(np.linalg.det(Y)) / np.power(np.abs(np.linalg.det(Y)), (1.0 / 3.0))) * Y
        [u, s, v] = np.linalg.svd(Y, full_matrices=False)
        Y = np.matmul(u, v)

        A = np.zeros((3*n, 6))
        b = np.zeros((3*n, 1))
        for i in range(n):
            A[3*i: 3*(i+1), :] = np.hstack((-AA[0:3, 4*i:4*i+3], np.eye(3)))
            b[3*i: 3*(i+1), :] = AA[0:3, 4*i+3, np.newaxis] - np.matmul(np.kron(BB[0:3, 4*i+3].T, np.eye(3)), np.reshape(Y, (9, 1)))

        t = np.linalg.lstsq(A, b, rcond=None)

        X = np.hstack((X, t[0][0:3]))
        X = np.vstack((X, np.asarray([[0, 0, 0, 1]])))
        Y = np.hstack((Y, t[0][3:6]))
        Y = np.vstack((Y, np.asarray([[0, 0, 0, 1]])))

        self.X = X
        self.Y = Y

        return X, Y





    def test_functions(self):
        images = 10
        target_x_transform = np.zeros((4,4*images))
        target_x_y_transform = np.zeros((4, 4 * images))
        robot_transform = np.zeros((4,4*images))
        the_X = np.asarray([[1, 0, 0, 0.1], [0, 1, 0, 0.1], [0, 0, 1, 0.1], [0, 0, 0, 1]])
        the_X_inverse = np.asarray([[1, 0, 0, -0.1], [0, 1, 0, -0.1], [0, 0, 1, -0.1], [0, 0, 0, 1]])
        the_Y_inverse = np.asarray([[1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, -1], [0, 0, 0, 1]])


        for i in range(images):
            #M = matrix( [rnd.random(), rnd.random(), rnd.random()], [rnd.random(), rnd.random(), rnd.random()])
            M = matrix(np.asarray([0.1*(i + 1), 0.1*(i + 1), 0.1])*(180.0/np.pi), [0.1, 0.1, 0.1])
            robot_transform[:, i * 4: (i + 1) * 4] = np.asarray(M)
            target_x_transform[:, i * 4: (i + 1) * 4] = np.matmul(np.matmul(the_X_inverse, M), the_X)
            target_x_y_transform[:, i * 4: (i + 1) * 4] = np.matmul(np.matmul(the_Y_inverse, M), the_X)

        test_1 = self.robot_arm_to_target_coupled( robot_transform, target_x_transform)
        test_2 = self.robot_arm_to_target_decoupled( robot_transform, target_x_transform)
        test_3 = self.robot_arm_target_world_decoupled(robot_transform, target_x_y_transform)
        return test_1

def trig(angle):
    r = radians(angle)
    return cos(r), sin(r)

def matrix(rotation, translation):
    xC, xS = trig(rotation[0])
    yC, yS = trig(rotation[1])
    zC, zS = trig(rotation[2])
    dX = translation[0]
    dY = translation[1]
    dZ = translation[2]
    return [[yC*xC, -zC*xS+zS*yS*xC, zS*xS+zC*yS*xC, dX],
            [yC*xS, zC*xC+zS*yS*xS, -zS*xC+zC*yS*xS, dY],
            [-yS, zS*yC, zC*yC, dZ],
            [0, 0, 0, 1]]

