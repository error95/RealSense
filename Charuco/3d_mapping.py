#Import libaries to use the RealSense cameas Charuco boards and rendering software
import pyrealsense2 as rs
import numpy as np
from cv2 import aruco
import helper_functions_charuco as hf
import calibration_kabsch_charuco as cc
import time
import open3d as o3d
import math



# Import helper functions and classes written to wrap the RealSense, OpenCV and Kabsch Calibration usage
from realsense_device_manager import DeviceManager, post_process_depth_frame
from stitcher_functions import get_transformation_matrix_wout_rsobject, get_charuco_points, get_charuco_points_ID
from calibration_kabsch_charuco import PoseEstimation
from calibration_kabsch_charuco import Transformation
from helper_functions_charuco import calculate_rmsd
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
    starting_points_found = False

    while not starting_points_found:
        frames = device_manager.poll_frames()
        corners = get_corners(device_list[0], frames, intrinsics_devices,charuco_board)
        print("Show camera charuco board")
        time.sleep(1)
        if all(corners[3]):
            starting_points = corners
            starting_points_found = True
    print("Charuco board found")
    device_manager.enable_emitter(True)
    key_list = device_manager.poll_frames().keys()
    pcd = o3d.geometry.PointCloud()

    # enable visualiser
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    first_image = True
    visualised_cloud = np.transpose(np.array([[0, 0, 0], [0, 0, 0]]))
    last_used = time.clock()
    prev_transform = [cc.Transformation(np.eye(3), np.array([0, 0, 0])), 0]
    while True:
        if first_image:
            time.sleep(1)
        corners_new_point = get_corners(device_list[0], frames, intrinsics_devices, charuco_board)

        if time.clock() - last_used > 2 or first_image:
            print("Starting imaging")
            from_camera = np.transpose(np.array([0, 0, 0]))
            to_camera = np.transpose(np.array([0, 0, 0]))
            frames = device_manager.poll_frames()
            for corner in range(len(charuco_board.nearestMarkerCorners)):
                if starting_points[3][corner] and corners_new_point[3][corner]:
                    to_camera = np.vstack((to_camera, np.array(starting_points[2][:, corner])))
                    from_camera = np.vstack((from_camera, np.array(corners_new_point[2][:, corner])))
            if np.size(to_camera) > 25:
                print("update")
                last_used = time.clock()
                transformation = get_transform_short(from_camera, to_camera)
                difference = np.sum(np.absolute(transformation[0].pose_mat - prev_transform[0].pose_mat))
                if difference < 0.1:


                    start = time.time()
                    the_frames = device_manager.poll_frames()


                    frame = frames[device_list[0]][rs.stream.depth]
                    pc = rs.pointcloud()
                    points = pc.calculate(frame)
                    vert = np.transpose(np.asanyarray(points.get_vertices(2)))
                    calibrated_points = transformation[0].apply_transformation(vert)
                    cloud = calibrated_points
                    idxe = np.random.permutation(cloud.shape[1])

                    # Reduces rendered points and removes points with extreme z values
                    keep_ratio = 0.01
                    cloud_filtered = cloud[:, idxe[0:math.floor(cloud.shape[1] * keep_ratio)]]
                    # cloud_filtered = cloud_filtered - np.min(cloud_filtered[2, :])
                    dist_thresh = 3
                    #cloud_filtered = -cloud_filtered[:, cloud_filtered[2, :] < dist_thresh]
                    visualised_cloud = np.hstack((visualised_cloud,cloud_filtered))
                    # cloud_filtered = cloud_filtered[:, cloud_filtered[2, :] > -1]
                    # cloud_filtered = cloud_filtered[:, np.invert(np.any(cloud_filtered > dist_thresh, axis=0))]
                    # cloud_filtered = cloud_filtered[:, np.invert(np.any(cloud_filtered > dist_thresh, axis=0))]

                    # renders points from all different cameras
                    # mlab.points3d( cloud_filtered[0, :],  cloud_filtered[1, :],  cloud_filtered[2, :], scale_factor=0.1)
                    # mlab.show()

                    pcd.points = o3d.utility.Vector3dVector(np.transpose(visualised_cloud))

                if first_image:
                    vis.add_geometry(pcd)
                    first_image = False
                vis.update_geometry()
                vis.poll_events()
                vis.update_renderer()
                prev_transform = transformation
        if not first_image:
            vis.update_geometry()
            vis.poll_events()
            vis.update_renderer()





    '''''''''
    Pseudo code:
    
    -Take imege to get starting position
    
    while true:
        get new positionfrom camera
        extract pointcloud from that position and transform it to starting position 
        
    
    '''''''''''
def get_corners(camera, camera_frame, intrinsic, charuco_board):
    ir_frame = camera_frame[camera][(rs.stream.infrared, 1)]
    depth_frame = post_process_depth_frame(camera_frame[camera][rs.stream.depth])
    depth_intrinsics = intrinsic[camera][rs.stream.depth]
    list_found_corners, IDs = hf.cv_find_chessboard(depth_frame, ir_frame, charuco_board)
    validPoints = [False] * charuco_board.chessboardCorners.shape[0]
    corners3D = [IDs, None, None, validPoints]
    points3D = np.zeros((3, charuco_board.chessboardCorners.shape[0]))
    if list_found_corners:
        found_corners = list_found_corners[0]
        for index in range(len(found_corners)):
            theID = IDs[0][index][0]
            corner = found_corners[index].flatten()
            depth = hf.get_depth_at_pixel(depth_frame, corner[0], corner[1])
            if depth != 0 and depth is not None:
                validPoints[theID] = True
                [X, Y, Z] = hf.convert_depth_pixel_to_metric_coordinate(depth, corner[0], corner[1],
                                                                     depth_intrinsics)
                points3D[0, theID] = X
                points3D[1, theID] = Y
                points3D[2, theID] = Z
        corners3D = IDs, found_corners, points3D, validPoints
    return corners3D

def get_transform_short(to_camera, from_camera):

    rotation, translation, rmsd = cc.calculate_transformation_kabsch(np.transpose(to_camera[1:, :]), np.transpose(from_camera[1:, :]))
    return Transformation(rotation, translation), rmsd


if __name__ == '__main__':
    main()