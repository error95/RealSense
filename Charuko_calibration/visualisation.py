#Import libaries to use the RealSense cameas Charuco boards and rendering software
import pyrealsense2 as rs
import numpy as np
from cv2 import aruco
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


def calculate_avg_rms_error(device_list, device_manager, transformation_devices, charuco_board, intrinsics_devices):
    while True:
        frames = device_manager.poll_frames()
        rmsd_list = np.full(len(device_list) - 1, np.inf)
        source_camera = device_list[0]

        source_points, IDs_source = get_charuco_points_ID(frames[device_list[0]],
                                                          transformation_devices[device_list[0]],
                                                          intrinsics_devices[device_list[0]], charuco_board)
        for idx, compared_camera in enumerate(device_list[1:]):

            compared_points, IDs_compared = get_charuco_points_ID(frames[compared_camera],
                                                                  transformation_devices[compared_camera],
                                                                  intrinsics_devices[compared_camera],
                                                                  charuco_board)
            validPoints = IDs_source & IDs_compared
            if np.any(validPoints):
                source_points_filtered = source_points[:, validPoints]
                compared_points = compared_points[:, validPoints]
                dist = source_points_filtered - compared_points
                rmsd = 0
                N = source_points_filtered.shape[1]
                for col in range(N):
                    rmsd += np.matmul(dist[:, col].transpose(), dist[:, col]).flatten()[0]
                rmsd_list[idx] = np.sqrt(rmsd / N)
        average = 0
        num = 0
        found_corners = False
        for x in rmsd_list:
            if x != np.inf:
                found_corners = True
                average += x
                num += 1
        if found_corners:
            print(average / num)


def visualise_chessboard(device_manager, device_list, intrinsics_devices, charuco_board, transformation_devices):
    # enable visualiser
    pcd = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    first_image = True

    while True:
        frames = device_manager.poll_frames()
        displayed_points = np.zeros((10, 3))
        for camera in device_list:
            added_points = get_charuco_points(frames[camera], transformation_devices[camera],
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


def visualise_point_cloud(key_list, resolution_height, resolution_width, device_manager, coordinate_dimentions, transformation_devices):
    # enable visualiser
    pcd = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    first_image = True

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
            test = pc.map_to(frame)
            vert = np.transpose(np.asanyarray(points.get_vertices(2)))
            transformed_pixels[:coordinate_dimentions, :] = vert
            calibrated_points = np.matmul(transformation_devices[camera].pose_mat, transformed_pixels)
            cloud[:, pixels * idx: pixels * (idx + 1)] = calibrated_points[:coordinate_dimentions, :]

        # Reduces rendered points and removes points with extreme z values
        keep_ratio = 0.005
        cloud_filtered = cloud[:, idxe[0:math.floor(cloud.shape[1] * keep_ratio)]]
        #cloud_filtered = cloud_filtered - np.min(cloud_filtered[2, :])
        dist_thresh = 3
        cloud_filtered = -cloud_filtered[:, cloud_filtered[2, :] < dist_thresh]
        #cloud_filtered = cloud_filtered[:, cloud_filtered[2, :] > -1]
        #cloud_filtered = cloud_filtered[:, np.invert(np.any(cloud_filtered > dist_thresh, axis=0))]
        #cloud_filtered = cloud_filtered[:, np.invert(np.any(cloud_filtered > dist_thresh, axis=0))]

        # renders points from all different cameras
        # mlab.points3d( cloud_filtered[0, :],  cloud_filtered[1, :],  cloud_filtered[2, :], scale_factor=0.1)
        # mlab.show()

        pcd.points = o3d.utility.Vector3dVector(np.transpose(cloud_filtered))

        if first_image:
            vis.add_geometry(pcd)
            first_image = False
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()

def create_point_cloud(key_list, resolution_height, resolution_width, device_manager, coordinate_dimentions,
                          transformation_devices):
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
            calibrated_points = np.matmul(transformation_devices[camera].pose_mat, transformed_pixels)
            cloud[:, pixels * idx: pixels * (idx + 1)] = calibrated_points[:coordinate_dimentions, :]

        print(1/(start - time.time()))



def visualise_rgbd_cloud(key_list, resolution_height, resolution_width, device_manager, coordinate_dimentions, transformation_devices):
    # enable visualiser
    align = rs.align(rs.stream.depth)
    vis = o3d.Visualizer()
    vis.create_window('PCD', width=1280, height=720)
    pointcloud = o3d.PointCloud()
    geom_added = False
    voxel_size = 0.005

    while True:
        cloud_pcd = o3d.PointCloud()
        for (serial, device) in device_manager._enabled_devices.items():
            frames = device.pipeline.wait_for_frames()
            frames = align.process(frames)
            profile = frames.get_profile()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            img_depth = o3d.Image(depth_image)
            img_color = o3d.Image(color_image)
            rgbd = o3d.create_rgbd_image_from_color_and_depth(img_color, img_depth, convert_rgb_to_intensity=False)

            intrinsics = profile.as_video_stream_profile().get_intrinsics()
            pinhole_camera_intrinsic = o3d.PinholeCameraIntrinsic(intrinsics.width, intrinsics.height, intrinsics.fx,
                                                              intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
            pcd = o3d.create_point_cloud_from_rgbd_image(rgbd, pinhole_camera_intrinsic)

            pcd.transform(transformation_devices[serial].pose_mat)
            pointcloud.points = pcd.points
            pointcloud.colors = pcd.colors

            cloud_pcd = cloud_pcd + pointcloud

        #pcd = o3d.geometry.voxel_down_sample(cloud_pcd,
                                                          # voxel_size=voxel_size)
        while True:
            if geom_added == False:
                vis.add_geometry(cloud_pcd)
                geom_added = True

            vis.update_geometry()
            vis.poll_events()
            vis.update_renderer()



