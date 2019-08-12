
#Import libaries to use the RealSense cameas Charuco boards and rendering software
import pyrealsense2 as rs
import numpy as np
from cv2 import aruco
import time


# Import helper functions and classes written to wrap the RealSense, OpenCV and Kabsch Calibration usage
from realsense_device_manager import DeviceManager, post_process_depth_frame
from stitcher_functions import get_transformation_matrix_wout_rsobject
from calibration_kabsch_charuco import Transformation
from helper_functions_charuco import matrix_viewer, least_error_transfroms
from visualisation import calculate_avg_rms_error, visualise_chessboard,\
    visualise_point_cloud, visualise_rgbd_cloud, create_point_cloud


class Camera:
    """ ................................................................................................... "
    "                                                                                                       "
    "  Initializes a Camera object. If a Calibration file is received, it will use the save file            "
    "  to calibrate the cameras                                                                             "
    "                                                                                                       "
    "  Returns: none                                                                                        "
    "                                                                               Sindre Skaar            "
    "                                                                                                       "
    " ..................................................................................................  """

    def __init__(self, init_file_camera=None):
        self.resolution_width = 1280  # pixels
        self.resolution_height = 720  # pixels
        self.frame_rate = 30  # fps
        dispose_frames_for_stablisation = 30  # frames
        self.coordinate_dimentions = 3
        self.cameras_calibrated = False
        self.charuco_boards = {}
        self.transformation_devices = {}
        # Enable the streams from all the intel realsense devices
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.depth, self.resolution_width, self.resolution_height, rs.format.z16, self.frame_rate)
        rs_config.enable_stream(rs.stream.infrared, 1, self.resolution_width, self.resolution_height, rs.format.y8, self.frame_rate)
        rs_config.enable_stream(rs.stream.color, self.resolution_width, self.resolution_height, rs.format.bgr8, self.frame_rate)

        # Use the device manager class to enable the devices and get the frames
        self.device_manager = DeviceManager(rs.context(), rs_config)
        self.device_manager.enable_all_devices()
        self.device_manager._available_devices.sort()
        self.device_list = self.device_manager._available_devices


        #initialize pointcloud depth image matrix
        self.pcs = {}
        for camera in self.device_list:
            self.pcs[camera] = rs.pointcloud()
        self.pixels = self.resolution_width * self.resolution_height
        self.total_pixels = self.pixels * len(self.device_list)
        self.cloud = np.zeros((3, self.total_pixels))
        self.transformed_pixels = np.ones((4, self.pixels))


        # Allow some frames for the auto-exposure controller to stablise
        for frame in range(dispose_frames_for_stablisation):
            frames = self.device_manager.poll_frames_keep()

        assert (len(self.device_manager._available_devices) > 0)

        # Then we define the charuco boards used in the rig
        self.set_charuco()

        # Get the intrinsics of the realsense device
        self.intrinsics_devices = self.device_manager.get_device_intrinsics(frames)

        try:
            transfer_matirices_save = np.load(init_file_camera, allow_pickle=True)
            transfer_matrices = transfer_matirices_save[()]
            correct_filename = True
            transformation_devices = {}
            for matrix in transfer_matrices:
                the_matrix = transfer_matrices[matrix]
                transformation_devices[matrix] = Transformation(the_matrix[:3, :3], the_matrix[:3, 3])
            self.cameras_calibrated = True

            self.transformation_devices = transformation_devices
        except:
            print('No such file in directory: "', init_file_camera, '"')
            print("Rig not calibrated\n")
            return

        print("Calibration completed...\n")
        return

    def set_charuco(self):
        """ ................................................................................................... "
        "                                                                                                       "
        "   Creates the Charuco boards that will be used during the experiment.                                 "
        "   All Charuco boards made are saved in the class within self.charuco_boards dict.                     "
        "                                                                                                       "
        "   Returns: None                                                                                       "
        "                                                                                                       "
        "                                                                               Sindre Skaar            "
        "                                                                                                       "
        " ..................................................................................................  """
        charuco_boards = {}

        # Set the charuco board parameters for calibration
        aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)
        charuco_width = 8
        charuco_height = 5
        square_length = 0.0392
        marker_length = square_length*0.8

        charuco_boards['charuco_board'] = aruco.CharucoBoard_create(charuco_width, charuco_height, square_length,
                                                                    marker_length,
                                                                    aruco_dict)
        # Set the charuco board parameters for robot
        aruco_dict_r = aruco.Dictionary_get(aruco.DICT_4X4_250)
        charuco_width_r = 3
        charuco_height_r = 3
        square_length_r = 0.0311
        marker_length_r = .0247

        charuco_boards['charuco_robot'] = aruco.CharucoBoard_create(charuco_width_r, charuco_height_r,
                                                                          square_length_r,
                                                                          marker_length_r, aruco_dict_r)

        # Set the charuco board parameters for robot
        aruco_dict_ro = aruco.Dictionary_get(aruco.DICT_5X5_100)
        charuco_width_ro = 3
        charuco_height_ro = 3
        square_length_ro = 0.0311
        marker_length_ro = .0247

        charuco_boards['charuco_target'] = aruco.CharucoBoard_create(charuco_width_ro, charuco_height_ro,
                                                                            square_length_ro,
                                                                            marker_length_ro,
                                                                            aruco_dict_ro)
        self.charuco_boards = charuco_boards

    def calibrate(self, amount_frames=150):
        """ ................................................................................................... "
        "                                                                                                       "
        "  Calibrates the cameras by creating a transformation matrix between the different cameras.            "
        "  The calibration will continue until all cameras has a transformation matrix to the camera with       "
        "  the lowest serial number. The function will not return if it cannot find all matrices                "
        "                                                                                                       "
        "  Returns: transformation_devices, A dict containing all transfer_matrces from al cameras
        "  to lowest serial number camera. The key in the dict is the serial number in the "from" camera
        "                                                                                                       "
        "                                                                               Sindre Skaar            "
        "                                                                                                       "
        " ..................................................................................................  """
        # Make a dict to store all images for calibration
        frame_dict = {}
        transform_dict = {}
        rms_dict = {}
        for from_device in self.device_list:
            transform_dict[from_device] = {}
            rms_dict[from_device] = {}
            for to_device in self.device_list:
                transform_dict[from_device][to_device] = {}
                rms_dict[from_device][to_device] = np.inf
        print("Starting to take images in 5 seconds")
        time.sleep(5)
        devices_stitched = False

        while not devices_stitched:
            print("taking new set of  images")
            for frame_count in range(amount_frames):
                print("taking image")
                print(amount_frames - frame_count, "images left")
                frames = self.device_manager.poll_frames_keep()
                time.sleep(0.5)
                frame_dict[frame_count] = {}
                for device in self.device_list:
                    ir_frame = np.asanyarray(frames[device][(rs.stream.infrared, 1)].get_data())
                    depth_frame = np.asanyarray(
                        post_process_depth_frame(frames[device][rs.stream.depth]).get_data())
                    frame_dict[frame_count][device] = {'ir_frame': ir_frame, 'depth_frame': depth_frame}
                del frames

            # Make transfer matrices between all possible cameras
            for idx, from_device in enumerate(self.device_list[:-1]):
                for to_device in self.device_list[idx + 1:]:
                    if to_device != from_device:
                        temp_transform, temp_rms = get_transformation_matrix_wout_rsobject(frame_dict,
                                                                                           [from_device, to_device],
                                                                                           self.intrinsics_devices,
                                                                                           self.charuco_boards['charuco_board'])
                        if temp_rms < rms_dict[from_device][to_device]:
                            rms_dict[from_device][to_device] = temp_rms
                            rms_dict[to_device][from_device] = temp_rms
                            transform_dict[from_device][to_device] = temp_transform
                            transform_dict[to_device][from_device] = temp_transform.inverse()

            # Use Dijkstra to find shortest path and check if all cameras are connected
            transformation_matrices = least_error_transfroms(transform_dict, rms_dict)
            if transformation_matrices != 0:
                devices_stitched = True
            # Prints rms error between camera transforms
            test = matrix_viewer(rms_dict)
            print(test)

        # Turns transformation matrices into Transfomation objects
        transformation_devices = {}
        for matrix in transformation_matrices:
            the_matirx = transformation_matrices[matrix]
            transformation_devices[matrix] = Transformation(the_matirx[:3, :3], the_matirx[:3, 3])

        self.transformation_devices = transformation_devices

        save_calibration = input('Press "y" if you want to save calibration \n')
        if save_calibration == "y":
            saved = False
            while not saved:
                name = input("Type in name of file to save. remember to end name with '.npy' \n")
                try:
                    np.save(name, transformation_matrices)
                    saved = True
                except:
                    print("could not save, try another name and remember '.npy' in the end")

        frame_dict.clear()
        self.cameras_calibrated = True
        return self.transformation_devices

    def visualise(self):
        """ ................................................................................................... "
        "                                                                                                       "
        "  A functions used to easily visualise the errors, by calculating the rms errors on a charuco board,   "
        "  visualising the chess board points, visualising point cloud stream or an rgbd image.                 "
        "  all functions are made as infinite while loops and are thus, just for visualisation and does not     "
        "  return anything.                                                                                     "
        "                                                                                                       "
        "  Returns: None                                                                                        "
        "                                                                                                       "
        "                                                                               Sindre Skaar            "
        "                                                                                                       "
        " ..................................................................................................  """
        if not self.cameras_calibrated:
            print("Cameras not calibrated")
            return
        self.device_manager.enable_emitter(True)
        key_list = self.device_manager.poll_frames().keys()

        while True:
            visualisation = input('Presss "1" for RMS error, "2" for chessboard visualisation and "3" for 3d pointcloud and "4" for robot to camea calibration\n')

            if visualisation == '1':

                calculate_avg_rms_error(self.device_list, self.device_manager, self.transformation_devices, self.charuco_boards['charuco_board'],
                                        self.intrinsics_devices)

            elif visualisation == '2':

                visualise_chessboard(self.device_manager, self.device_list, self.intrinsics_devices, self.transformation_devices['charuco_board'],
                                     self.transformation_devices)

            elif visualisation == '3':

                visualise_point_cloud(key_list, self.resolution_height, self.resolution_width, self.device_manager,
                                      self.coordinate_dimentions,
                                      self.transformation_devices)
            elif visualisation == '4':
                visualise_rgbd_cloud(key_list, self.resolution_height, self.resolution_width, self.device_manager,
                                      self.coordinate_dimentions,
                                      self.transformation_devices)
            elif visualisation == '5':
                create_point_cloud(key_list, self.resolution_height, self.resolution_width, self.device_manager,
                                      self.coordinate_dimentions,
                                      self.transformation_devices)

            else:
                print("Input not recognised")

    def get_robot_data(self):
        """Returns data necessary for the robot calibration"""

        return self.transformation_devices, self.device_manager,\
               self.charuco_boards['charuco_target'], self.charuco_boards['charuco_robot']

    def get_transform_matrix(self):
        """Returns the transfer matrices used to go from all cameras to lowest serial camera.
           Returns -1 if the rig is not calibrated"""

        if self.cameras_calibrated:
            return self.transformation_devices
        else:
            print('Cameras not calibrated')
        return -1

    def poll_depth_frame(self):

        """ ................................................................................................... "
        "                                                                                                       "
        "  Polls the depth frames form all connected cameras, transforms the point clouds to lowest             "
        "  serial number cameras coordinate system and concatenates them together. If rig is not calibrated     "
        "  it returns -1                                                                                        "
        "                                                                                                       "
        "  Returns: cloud,  a point cloud form all cameras stitched together into one point cloud.              "
        "  Or returns -1 if cameras are not calibrated                                                          "
        "                                                                                                       "
        "                                                                                                       "
        "                                                                               Sindre Skaar            "
        "                                                                                                       "
        " ..................................................................................................  """

        if not self.cameras_calibrated:
            print("Cameras not calibrated")
            return -1

        the_frames = self.device_manager.poll_frames()

        for idx, camera in enumerate(self.device_list):
            pc = self.pcs[camera]
            frame = the_frames[camera][rs.stream.depth]
            points = pc.calculate(frame)
            vert = np.transpose(np.asanyarray(points.get_vertices(2)))
            self.transformed_pixels[:3, :] = vert
            calibrated_points = np.matmul(self.transformation_devices[camera].pose_mat, self.transformed_pixels)
            self.cloud[:, self.pixels * idx: self.pixels * (idx + 1)] = calibrated_points[:3, :]
        return self.cloud
