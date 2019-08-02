
import pyrealsense2 as rs

import numpy as np

from helper_functions_charuco import cv_find_chessboard, calculate_rmsd, get_depth_at_pixel, convert_depth_pixel_to_metric_coordinate, cv_find_chessboard_np
from realsense_device_manager import post_process_depth_frame
from calibration_kabsch_charuco import calculate_transformation_kabsch, Transformation


def get_3d_points(frames_dict, camera_set, intrinsic, charuco_board):
    from_camera = np.transpose(np.array([0, 0, 0]))
    to_camera = np.transpose(np.array([0, 0, 0]))
    for frame_count in range(len(frames_dict)):
        camera_frame = frames_dict
        corners3D = {}
        for camera in camera_set:
            ir_frame = camera_frame[camera][(rs.stream.infrared, 1)]
            depth_frame = post_process_depth_frame(camera_frame[camera][rs.stream.depth])
            depth_intrinsics = intrinsic[camera][rs.stream.depth]
            list_found_corners, IDs = cv_find_chessboard(depth_frame, ir_frame, charuco_board)
            validPoints = [False] * charuco_board.chessboardCorners.shape[0]
            corners3D[camera] = [IDs, None, None, validPoints]
            points3D = np.zeros((3, charuco_board.chessboardCorners.shape[0]))
            if list_found_corners:
                found_corners = list_found_corners[0]
                for index in range(len(found_corners)):
                    theID = IDs[0][index][0]
                    corner = found_corners[index].flatten()
                    depth = get_depth_at_pixel(depth_frame, corner[0], corner[1])
                    if depth != 0 and depth is not None:
                        validPoints[theID] = True
                        [X, Y, Z] = convert_depth_pixel_to_metric_coordinate(depth, corner[0], corner[1],
                                                                             depth_intrinsics)
                        points3D[0, theID] = X
                        points3D[1, theID] = Y
                        points3D[2, theID] = Z
                corners3D[camera] = IDs, found_corners, points3D, validPoints
        for corner in range(len(charuco_board.nearestMarkerCorners)):
            if corners3D[camera_set[0]][3][corner] and corners3D[camera_set[1]][3][corner]:
                to_camera = np.vstack((to_camera, np.array(corners3D[camera_set[0]][2][:, corner])))
                from_camera = np.vstack((from_camera, np.array(corners3D[camera_set[1]][2][:, corner])))
    return from_camera, to_camera


def get_3d_points_wout_rsobject(frames_dict, camera_set, intrinsic, charuco_board):
    from_camera = np.transpose(np.array([0, 0, 0]))
    to_camera = np.transpose(np.array([0, 0, 0]))
    for frame_count in range(len(frames_dict)):
        camera_frame = frames_dict[frame_count]
        corners3D = {}
        for camera in camera_set:
            ir_frame = camera_frame[camera]['ir_frame']
            depth_frame = post_process_depth_frame(camera_frame[camera]['depth_frame'])
            depth_intrinsics = intrinsic[camera][rs.stream.depth]
            list_found_corners, IDs = cv_find_chessboard(depth_frame, ir_frame, charuco_board)
            validPoints = [False] * charuco_board.chessboardCorners.shape[0]
            corners3D[camera] = [IDs, None, None, validPoints]
            points3D = np.zeros((3, charuco_board.chessboardCorners.shape[0]))
            if list_found_corners:
                found_corners = list_found_corners[0]
                for index in range(len(found_corners)):
                    theID = IDs[0][index][0]
                    corner = found_corners[index].flatten()
                    depth = get_depth_at_pixel(depth_frame, corner[0], corner[1])
                    if depth != 0 and depth is not None:
                        validPoints[theID] = True
                        [X, Y, Z] = convert_depth_pixel_to_metric_coordinate(depth, corner[0], corner[1],
                                                                             depth_intrinsics)
                        points3D[0, theID] = X
                        points3D[1, theID] = Y
                        points3D[2, theID] = Z
                corners3D[camera] = IDs, found_corners, points3D, validPoints
        for corner in range(len(charuco_board.nearestMarkerCorners)):
            if corners3D[camera_set[0]][3][corner] and corners3D[camera_set[1]][3][corner]:
                to_camera = np.vstack((to_camera, np.array(corners3D[camera_set[0]][2][:, corner])))
                from_camera = np.vstack((from_camera, np.array(corners3D[camera_set[1]][2][:, corner])))
    return from_camera, to_camera

def get_transform_short(to_camera, from_camera):
    if np.size(from_camera) > 30:
        rotation, translation, rmsd = calculate_transformation_kabsch(np.transpose(to_camera[1:, :]), np.transpose(from_camera[1:, :]))
        return Transformation(rotation, translation), rmsd

    else:
        return 0, np.inf


def get_transformation_matrix(frames_dict, camera_set, intrinsic, charuco_board):
    from_camera = np.transpose(np.array([0, 0, 0]))
    to_camera = np.transpose(np.array([0, 0, 0]))
    for frame_count in range(len(frames_dict)):
        camera_frame = frames_dict[frame_count]
        corners3D = {}
        for camera in camera_set:
            ir_frame = camera_frame[camera][(rs.stream.infrared, 1)]
            depth_frame = post_process_depth_frame(camera_frame[camera][rs.stream.depth])
            depth_intrinsics = intrinsic[camera][rs.stream.depth]
            list_found_corners, IDs = cv_find_chessboard(depth_frame, ir_frame, charuco_board)
            validPoints = [False] * charuco_board.chessboardCorners.shape[0]
            corners3D[camera] = [IDs, None, None, validPoints]
            points3D = np.zeros((3, charuco_board.chessboardCorners.shape[0]))
            if list_found_corners:
                found_corners = list_found_corners[0]
                for index in range(len(found_corners)):
                    theID = IDs[0][index][0]
                    corner = found_corners[index].flatten()
                    depth = get_depth_at_pixel(depth_frame, corner[0], corner[1])
                    if depth != 0 and depth is not None:
                        validPoints[theID] = True
                        [X, Y, Z] = convert_depth_pixel_to_metric_coordinate(depth, corner[0], corner[1], depth_intrinsics)
                        points3D[0, theID] = X
                        points3D[1, theID] = Y
                        points3D[2, theID] = Z
                corners3D[camera] = IDs, found_corners, points3D, validPoints
        for corner in range(len(charuco_board.nearestMarkerCorners)):
            if corners3D[camera_set[0]][3][corner] and corners3D[camera_set[1]][3][corner]:
                to_camera = np.vstack((to_camera, np.array(corners3D[camera_set[0]][2][:, corner])))
                from_camera = np.vstack((from_camera, np.array(corners3D[camera_set[1]][2][:, corner])))
    if np.size(from_camera) > 30:
        rotation, translation, rmsd = calculate_transformation_kabsch(np.transpose(to_camera[1:, :]), np.transpose(from_camera[1:, :]))
        return Transformation(rotation, translation), rmsd

    else:
        return 0, np.inf


def get_transformation_matrix_wout_rsobject(frames_dict, camera_set, intrinsic, charuco_board):
    from_camera = np.transpose(np.array([0, 0, 0]))
    to_camera = np.transpose(np.array([0, 0, 0]))
    for frame_count in range(len(frames_dict)):
        camera_frame = frames_dict[frame_count]
        corners3D = {}
        for camera in camera_set:
            ir_frame = camera_frame[camera]['ir_frame']
            depth_frame = camera_frame[camera]['depth_frame']
            depth_intrinsics = intrinsic[camera][rs.stream.depth]
            list_found_corners, IDs = cv_find_chessboard_np(depth_frame, ir_frame, charuco_board)
            validPoints = [False] * charuco_board.chessboardCorners.shape[0]
            corners3D[camera] = [IDs, None, None, validPoints]
            points3D = np.zeros((3, charuco_board.chessboardCorners.shape[0]))
            if list_found_corners:
                found_corners = list_found_corners[0]
                for index in range(len(found_corners)):
                    theID = IDs[0][index][0]
                    corner = found_corners[index].flatten()
                    depth = depth_frame[int(round(corner[1])), int(round(corner[0]))]/1000
                    depth = subpixel_depth(depth_frame, corner[1], corner[0])
                    if depth != 0 and depth is not None:
                        validPoints[theID] = True
                        [X, Y, Z] = convert_depth_pixel_to_metric_coordinate(depth, corner[0], corner[1], depth_intrinsics)
                        points3D[0, theID] = X
                        points3D[1, theID] = Y
                        points3D[2, theID] = Z
                corners3D[camera] = IDs, found_corners, points3D, validPoints
        for corner in range(len(charuco_board.nearestMarkerCorners)):
            if corners3D[camera_set[0]][3][corner] and corners3D[camera_set[1]][3][corner]:
                to_camera = np.vstack((to_camera, np.array(corners3D[camera_set[0]][2][:, corner])))
                from_camera = np.vstack((from_camera, np.array(corners3D[camera_set[1]][2][:, corner])))
    if np.size(from_camera) > 90:
        rotation, translation, rmsd = calculate_transformation_kabsch(np.transpose(to_camera[1:, :]), np.transpose(from_camera[1:, :]))
        print(np.size(from_camera), 'points found. TRANSFORM MATRIX MADE')
        return Transformation(rotation, translation), rmsd

    else:
        print("Only", np.size(from_camera), 'points found. COMPARISON DUMPED')
        return 0, np.inf















def get_transfrom(A, B):

    dimension, points = np.shape(A)

    A_centre = np.mean(A, axis=1)
    B_centre = np.mean(B, axis=1)

    test = np.transpose(np.array([A_centre,]*points))
    A_corrected = A - np.transpose(np.array([A_centre,]*points))
    B_corrected = B - np.transpose(np.array([B_centre,]*points))

    A_B = np.matmul(A, np.transpose(B))
    u, s, v = np.linalg.svd(A_B)

    v_u = np.matmul(u, np.transpose(u))
    det_v_u = np.linalg.det(v_u)

    rotation_temp = np.matmul(v, np.diag([1, 1, det_v_u]))
    rotation = np.matmul(rotation_temp, np.transpose(u))

    transformation = B_centre - np.matmul(rotation, A_centre)

    return rotation, transformation

def get_charuco_points(camera_frame, transform, intrinsic, charuco_board):
    ir_frame = camera_frame[(rs.stream.infrared, 1)]
    depth_frame = post_process_depth_frame(camera_frame[rs.stream.depth])
    depth_intrinsics = intrinsic[rs.stream.depth]
    list_found_corners, IDs = cv_find_chessboard(depth_frame, ir_frame, charuco_board)
    board_points = np.array([0, 0, 0])
    if list_found_corners:
        found_corners = list_found_corners[0]
        for idx in range(len(found_corners)):
            corner = found_corners[idx].flatten()
            depth = get_depth_at_pixel(depth_frame, corner[0], corner[1])
            coords = convert_depth_pixel_to_metric_coordinate(depth, corner[0], corner[1], depth_intrinsics)
            board_points = np.vstack((board_points,coords))


        return transform.apply_transformation(np.transpose(board_points[1:, :]))
    else:
        return np.array([])

def get_charuco_points_ID(camera_frame, transform, intrinsic, charuco_board):
    ir_frame = camera_frame[(rs.stream.infrared, 1)]
    depth_frame = post_process_depth_frame(camera_frame[rs.stream.depth])
    depth_intrinsics = intrinsic[rs.stream.depth]
    list_found_corners, IDs = cv_find_chessboard(depth_frame, ir_frame, charuco_board)
    board_points = np.array(np.zeros((3, charuco_board.chessboardCorners.shape[0])))
    validPoints = np.array([False] * charuco_board.chessboardCorners.shape[0])
    if list_found_corners:
        found_corners = list_found_corners[0]
        for idx in range(len(found_corners)):
            corner = found_corners[idx].flatten()
            ID = IDs[0][idx][0]
            validPoints[ID] = True
            depth = get_depth_at_pixel(depth_frame, corner[0], corner[1])
            coords = convert_depth_pixel_to_metric_coordinate(depth, corner[0], corner[1], depth_intrinsics)
            board_points[:, ID] = coords

        return transform.apply_transformation(board_points), validPoints
    else:
        return np.array([]), np.array([False] * charuco_board.chessboardCorners.shape[0])




def get_full_transformation_matrix_wout_rsobject(frames_dict, camera_set, intrinsic, charuco_board):
    camera_points = np.transpose(np.array([0, 0, 0]))
    for frame_count in range(len(frames_dict)):
        camera_frame = frames_dict[frame_count]
        corners3D = {}
        for camera in camera_set:
            ir_frame = camera_frame[camera]['ir_frame']
            depth_frame = post_process_depth_frame(camera_frame[camera]['depth_frame'])
            depth_intrinsics = intrinsic[camera][rs.stream.depth]
            list_found_corners, IDs = cv_find_chessboard(depth_frame, ir_frame, charuco_board)
            validPoints = [False] * charuco_board.chessboardCorners.shape[0]
            corners3D[camera] = [IDs, None, None, validPoints]
            points3D = np.zeros((3, charuco_board.chessboardCorners.shape[0]))
            if list_found_corners:
                found_corners = list_found_corners[0]
                for index in range(len(found_corners)):
                    theID = IDs[0][index][0]
                    corner = found_corners[index].flatten()
                    depth = get_depth_at_pixel(depth_frame, corner[0], corner[1])
                    if depth != 0 and depth is not None:
                        validPoints[theID] = True
                        [X, Y, Z] = convert_depth_pixel_to_metric_coordinate(depth, corner[0], corner[1], depth_intrinsics)
                        points3D[0, theID] = X
                        points3D[1, theID] = Y
                        points3D[2, theID] = Z
                corners3D[camera] = IDs, found_corners, points3D, validPoints

        #if corners3D[from_camera][3][corner] and corners3D[to_camera][3][corner]:
        #    from_camera_points = np.vstack((from_camera_points, np.array(corners3D[from_camera][2][:, corner])))
        #    to_camera_points = np.vstack((to_camera_points, np.array(corners3D[to_camera][2][:, corner])))


    for idx, from_camera in camera_set:
        for to_camera in camera_set[idx + 1:]:
            for corner in range(len(charuco_board.nearestMarkerCorners)):
                if np.size(from_camera) > 30:
                   # rotation, translation, rmsd = calculate_transformation_kabsch(np.transpose(to_camera_points[1:, :]),
                     #                                                             np.transpose(from_camera_points[1:, :])
                    print("per")



def subpixel_depth(depth_frame, x, y):
    x_floor = int(np.floor(x))
    x_ceil = int(np.ceil(x))
    y_floor = int(np.floor(y))
    y_ceil = int(np.ceil(y))

    delta_x = x - x_floor
    delta_y = y - y_floor

    depth = depth_frame[x_floor, y_floor]*(1 - delta_x - delta_y) + delta_x*depth_frame[x_ceil, y_floor] + delta_y*depth_frame[x_floor, y_ceil]
    return depth/1000