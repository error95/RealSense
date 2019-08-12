#Import libaries to use the RealSense cameas Charuco boards and rendering software
import pyrealsense2 as rs
import numpy as np
from cv2 import aruco
import time
import open3d as o3d
import math



# Import helper functions and classes written to wrap the RealSense, OpenCV and Kabsch Calibration usage
from realsense_device_manager import DeviceManager
from stitcher_functions import get_transformation_matrix, get_charuco_points
from calibration_kabsch_charuco import PoseEstimation
from calibration_kabsch_charuco import Transformation
from helper_functions_charuco import least_error_transfroms
from helper_functions_charuco import matrix_viewer, least_error_transfroms



def main():

    # First we set up the cameras to start streaming
    # Define some constants
    resolution_width = 1280  # pixels
    resolution_height = 720  # pixels
    frame_rate = 30  # fps
    dispose_frames_for_stablisation = 30  # frames



    # Enable the streams from all the intel realsense devices
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
    rs_config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, frame_rate)
    rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)

    # Use the device manager class to enable the devices and get the frames
    device_manager = DeviceManager(rs.context(), rs_config)
    device_manager.enable_all_devices()
    device_manager._available_devices.sort()
    device_list = device_manager._available_devices

    # Allow some frames for the auto-exposure controller to stablise
    for frame in range(dispose_frames_for_stablisation):
        frames = device_manager.poll_frames()

    assert (len(device_manager._available_devices) > 0)





    #Then we calibrate the images

    # Get the intrinsics of the realsense device
    intrinsics_devices = device_manager.get_device_intrinsics(frames)

    # Set the charuco board parameters for calibration
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    charuco_width = 8
    charuco_height = 5
    square_length = 0.03425
    marker_length = .026

    coordinate_dimentions = 3

    charuco_board = aruco.CharucoBoard_create(charuco_width, charuco_height, square_length, marker_length, aruco_dict)

    # Choose amount of frames to average
    amount_frames = 12
    frame_dict = {}
    transform_dict = {}
    rms_dict = {}
    for from_device in device_list:
        transform_dict[from_device] = {}
        rms_dict[from_device] = {}
        for to_device in device_list:
            transform_dict[from_device][to_device] = {}
            rms_dict[from_device][to_device] = np.inf

    devices_stitched = False
    while not devices_stitched:
        print("taking new set of  images")
        for frame_count in range(amount_frames):
            print("taking image")
            print(amount_frames - frame_count, "images left")
            frames = device_manager.poll_frames()
            print("Next image in 1 seconds")
            time.sleep(1)
            frame_dict[frame_count] = frames


        for idx, from_device in enumerate(device_list[:-1]):
            for to_device in device_list[idx + 1:]:
                if to_device != from_device:
                    temp_transform, temp_rms = get_transformation_matrix(frame_dict, [from_device, to_device], intrinsics_devices, charuco_board)
                    if temp_rms < rms_dict[from_device][to_device]:
                        rms_dict[from_device][to_device] = temp_rms
                        rms_dict[to_device][from_device] = temp_rms
                        transform_dict[from_device][to_device] = temp_transform
                        transform_dict[to_device][from_device] = temp_transform.inverse()

        test = matrix_viewer(rms_dict)
        print(test)
        devices_stitched = True
        for idx, from_device in enumerate(device_list[1:]):
            if rms_dict[from_device][device_list[idx]] == np.inf:
                devices_stitched = False
    transformation_devices = {}
    identity = np.identity(4)
    transformation_devices[device_list[0]] = Transformation(identity[:3, :3], identity[:3, 3])
    for idx, from_device in enumerate(device_list[1:]):
        temp_transform = np.matmul(transformation_devices[device_list[idx]].pose_mat, transform_dict[from_device][device_list[idx]].pose_mat)
        transformation_devices[from_device] = Transformation(temp_transform[:3, :3], temp_transform[:3, 3])

    # Printing
    print("Calibration completed... \n")

    # Enable the emitter of the devices and extract serial numbers to identify cameras
    device_manager.enable_emitter(True)
    key_list = device_manager.poll_frames().keys()
    pcd = o3d.geometry.PointCloud()

    # enable visualiser
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    first_image = True

    while True:
        frames = device_manager.poll_frames()
        displayed_points = np.zeros((10,3))
        for camera in device_list:
            added_points= get_charuco_points(frames[camera], transformation_devices[camera],
                                     intrinsics_devices[camera], charuco_board)
            if added_points.any():
                displayed_points = np.vstack((displayed_points, np.transpose(added_points)))

        pcd.points = o3d.utility.Vector3dVector(displayed_points)
        if first_image:
            vis.add_geometry(pcd)
            first_image = False
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()

    device_manager.disable_streams()
    vis.destroy_window()










main()