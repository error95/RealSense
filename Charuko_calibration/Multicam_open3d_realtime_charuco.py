#Import libaries to use the RealSense cameas Charuco boards and rendering software
import pyrealsense2 as rs
import numpy as np
from cv2 import aruco
import time
import open3d as o3d
import math



# Import helper functions and classes written to wrap the RealSense, OpenCV and Kabsch Calibration usage
from realsense_device_manager import DeviceManager
from calibration_kabsch_charuco import PoseEstimation
from calibration_kabsch_charuco import Transformation
from helper_functions_charuco import least_error_transfroms
from helper_functions_charuco import matrix_viewer



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

    # Estimate the pose of the cameras compared to the first camera in the list
    amount_devices = len(device_manager._available_devices)
    transformation_matrix = {}
    rms_matrix = {}
    for device in device_manager._available_devices:
        transformation_matrix[device] = {}
        rms_matrix[device] = {}
        for device2 in device_manager._available_devices:
            rms_matrix[device][device2] = np.inf

    devices_stitched = False
    while not devices_stitched:
        frames = device_manager.poll_frames()
        pose_estimator = PoseEstimation(frames, intrinsics_devices, charuco_board)
        transformation_result_kabsch = pose_estimator.perform_pose_estimation()
        object_point = pose_estimator.get_chessboard_corners_in3d()
        calibrated_device_count = 0
        for device in device_manager._available_devices:
            if not transformation_result_kabsch[device][0]:
                print("Device", device, "needs to be calibrated")
            else:
                # If this is the first camera in the list
                if calibrated_device_count == 0:
                    source_matrix = transformation_result_kabsch[device][1]
                    source_device = device
                    source_rms = transformation_result_kabsch[device][3]
                else:
                    # If new estimate is better than previous
                    if source_rms + transformation_result_kabsch[device][3] < rms_matrix[device][source_device]:
                        rms_matrix[source_device][device] = source_rms + transformation_result_kabsch[device][3]
                        rms_matrix[device][source_device] = source_rms + transformation_result_kabsch[device][3]
                        slave_transfrom = transformation_result_kabsch[device][1].inverse()
                        multiplied_transform = np.matmul(source_matrix.pose_mat, slave_transfrom.pose_mat)
                        Multiplied_transform = Transformation(multiplied_transform[:3, :3], multiplied_transform[:3, 3])
                        transformation_matrix[device][source_device] = Multiplied_transform
                        temp_inverted_matrix = np.matmul(source_matrix.pose_mat, slave_transfrom.pose_mat)
                        inverted_transform = Transformation(temp_inverted_matrix[:3, :3], temp_inverted_matrix[:3, 3])
                        transformation_matrix[source_device][device] = inverted_transform.inverse()
                calibrated_device_count += 1

        # Check if all devices are stitched together
        transformation_devices = least_error_transfroms(transformation_matrix, rms_matrix)
        if transformation_devices != 0:
            devices_stitched = True
        test = matrix_viewer(rms_matrix)
        print(test)



    print("Calibration completed... \n")

    # Enable the emitter of the devices and extract serial numbers to identify cameras
    device_manager.enable_emitter(True)
    key_list = device_manager.poll_frames().keys()
    pcd = o3d.geometry.PointCloud()

    # enable visualiser
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    first_image = True

    #Stitch together all the different camera pointclouds from different cameras
    pcs = {}
    for camera in key_list:
        pcs[camera] = rs.pointcloud()
    pixels = resolution_width * resolution_height
    total_pixels = pixels * len(key_list)
    cloud = np.zeros((3, total_pixels))
    transformed_pixels = np.ones((4, pixels))
    idxe = np.random.permutation(cloud.shape[1])
    while True:
        start = time.time()
        the_frames = device_manager.poll_frames()

        for idx, camera in enumerate(key_list):
            pc = pcs[camera]
            frame = the_frames[camera][rs.stream.depth]
            points = pc.calculate(frame)
            vert = np.transpose(np.asanyarray(points.get_vertices(2)))
            transformed_pixels[:coordinate_dimentions, :] = vert
            calibrated_points = np.matmul(transformation_devices[camera], transformed_pixels)
            cloud[:, pixels * idx: pixels * (idx + 1)] = calibrated_points[:coordinate_dimentions, :]

        # Reduces rendered points and removes points with extreme z values
        keep_ratio = 0.01
        cloud_filtered = cloud[:, idxe[0:math.floor(cloud.shape[1] * keep_ratio)]]
        #cloud_filtered = cloud_filtered - np.min(cloud_filtered[2, :])
        dist_thresh = 3
        cloud_filtered = -cloud_filtered[:, cloud_filtered[2, :] < dist_thresh]
        # cloud_filtered = cloud_filtered[:, cloud_filtered[2, :] > -1]
        # cloud_filtered = cloud_filtered[:, np.invert(np.any(cloud_filtered > dist_thresh, axis=0))]
        # cloud_filtered = cloud_filtered[:, np.invert(np.any(cloud_filtered > dist_thresh, axis=0))]

        # renders points from all different cameras
        #mlab.points3d( cloud_filtered[0, :],  cloud_filtered[1, :],  cloud_filtered[2, :], scale_factor=0.1)
        #mlab.show()
        pcd.points = o3d.utility.Vector3dVector(np.transpose(cloud_filtered))
        if first_image:
            vis.add_geometry(pcd)
            first_image = False
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
        end = time.time()
        #txt = input()
        #print(1 / (end - start))

    device_manager.disable_streams()
    vis.destroy_window()


main()
