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
from visualisation import calculate_avg_rms_error, visualise_chessboard, visualise_point_cloud




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
        frames = device_manager.poll_frames_keep()

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


    # Decide if you want to use previous calibration or make a new one
    calibration_decision = input('To use previous calibration, press "y". For new calibration press any key \n')
    if calibration_decision != 'y':
        # Choose amount of frames to average
        correct_input = False
        while not correct_input:
            the_input = input("Write amount of frames you want to use to calibrate \n")
            try:
                amount_frames = int(the_input)
                if amount_frames > 0:
                    correct_input = True
                else:
                    print("Frame count has to be positive")
            except ValueError:
                print('Input has to be int')


        # Make a dict to store all images for calibration
        frame_dict = {}
        transform_dict = {}
        rms_dict = {}
        for from_device in device_list:
            transform_dict[from_device] = {}
            rms_dict[from_device] = {}
            for to_device in device_list:
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
                frames = device_manager.poll_frames()
                time.sleep(0.5)
                frame_dict[frame_count] = {}
                for device in device_list:
                    ir_frame = np.asanyarray(frames[device][(rs.stream.infrared, 1)].get_data())
                    depth_frame = np.asanyarray(post_process_depth_frame(frames[device][rs.stream.depth]).get_data())
                    frame_dict[frame_count][device] = {'ir_frame': ir_frame, 'depth_frame': depth_frame}
                del frames

            # Make transfer matrices between all possible cameras
            for idx, from_device in enumerate(device_list[:-1]):
                for to_device in device_list[idx + 1:]:
                    if to_device != from_device:
                        temp_transform, temp_rms = get_transformation_matrix_wout_rsobject(frame_dict, [from_device, to_device], intrinsics_devices, charuco_board)
                        if temp_rms < rms_dict[from_device][to_device]:
                            rms_dict[from_device][to_device] = temp_rms
                            rms_dict[to_device][from_device] = temp_rms
                            transform_dict[from_device][to_device] = temp_transform
                            transform_dict[to_device][from_device] = temp_transform.inverse()

            #Use Djikstra's to fin shortest path and check if all cameras are connected
            transformation_matrices = least_error_transfroms(transform_dict, rms_dict)
            if transformation_matrices != 0:
                devices_stitched = True
            # Prints rms error between camera transforms
            test = matrix_viewer(rms_dict)
            print(test)

        # Turns transformation matrices into Transfomation objects
        transformation_devices ={}
        for matrix in transformation_matrices:
            the_matirx = transformation_matrices[matrix]
            transformation_devices[matrix] = Transformation(the_matirx[:3,:3], the_matirx[:3,3])

        # Saves calibration transforms if the user wants to
        save_calibration = input('Press "y" if you want to save calibration \n')
        if save_calibration == "y":
            calibration_name = input
            saved = False
            while not saved:
                name = input("Type in name of fale to save. remember to end name with '.npy' \n")
                try:
                    np.save(name, transformation_matrices)
                    saved = True
                except:
                    print("could not save, try another name and remember '.npy' in the end")

        frame_dict.clear()
    else:
        correct_filename = False
        while not correct_filename:
            file_name = input('Type in calibration file name \n')
            try:
                transfer_matirices_save = np.load(file_name, allow_pickle=True)
                transfer_matrices = transfer_matirices_save[()]
                correct_filename = True
            except:
                print('No such file in directory: "', file_name, '"')
        transformation_devices = {}
        for matrix in transfer_matrices:
            the_matrix = transfer_matrices[matrix]
            transformation_devices[matrix] = Transformation(the_matrix[:3, :3], the_matrix[:3, 3])

    print("Calibration completed...")

    # Enable the emitter of the devices and extract serial numbers to identify cameras
    device_manager.enable_emitter(True)
    key_list = device_manager.poll_frames().keys()


    while True:
        visualisation = input('Presss "1" for RMS error, "2" for chessboard visualisation and "3" for 3d pointcloud\n')

        if visualisation == '1':

            calculate_avg_rms_error(device_list, device_manager, transformation_devices, charuco_board, intrinsics_devices)

        elif visualisation == '2':

            visualise_chessboard(device_manager, device_list, intrinsics_devices, charuco_board, transformation_devices)

        elif visualisation == '3':

            visualise_point_cloud(key_list, resolution_height, resolution_width, device_manager, coordinate_dimentions,
                                  transformation_devices)

        else:
             print("Input not recognised")



main()

# Claculating transfer matrices
#    devices_stitched = True
#   for idx, from_device in enumerate(device_list[1:]):
#       if rms_dict[from_device][device_list[idx]] == np.inf:
#           devices_stitched = False
# Worst case calculation
# transformation_devices = {}
# identity = np.identity(4)
# transformation_devices[device_list[0]] = Transformation(identity[:3, :3], identity[:3, 3])
# for idx, from_device in enumerate(device_list[1:]):
#   temp_transform = np.matmul(transformation_devices[device_list[idx]].pose_mat, transform_dict[from_device][device_list[idx]].pose_mat)
#   transformation_devices[from_device] = Transformation(temp_transform[:3, :3], temp_transform[:3, 3])
