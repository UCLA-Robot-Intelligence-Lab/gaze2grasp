# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import cv2
import json
import numpy as np
import sys
import torch
import os
import aria.sdk as aria
import time
from common import quit_keypress, update_iptables
from gaze_model.inference import infer
from projectaria_tools.core.mps import EyeGaze
from projectaria_tools.core.calibration import device_calibration_from_json_string, get_linear_camera_calibration
from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection
from projectaria_tools.core.sensor_data import ImageDataRecord
from projectaria_tools.core.sophus import SE3

from homography import HomographyManager # Import HomographyManager
from gaze_history import GazeHistory

from multicam import XarmEnv

from scipy.spatial.transform import Rotation as R
# from rs_streamer import RealsenseStreamer
from calib_utils.linalg_utils import transform


GRIPPER_SPEED, GRIPPER_FORCE, GRIPPER_MAX_WIDTH, GRIPPER_TOLERANCE = 0.1, 40, 0.08570, 0.01

serial_no = '317422074281'
# Get camera, load transforms, load robot 
# realsense_streamer = RealsenseStreamer(serial_no)

#transforms = np.load(f'calib/transforms_{serial_no}.npy', allow_pickle=True).item()
transforms = np.load(f'calib/transforms.npy', allow_pickle=True).item()
TCR = transforms[serial_no]['tcr']

robot = XarmEnv()


# file paths to model weights and configuration
model_weights = f"contact_graspnet/gaze_model/inference/model/pretrained_weights/social_eyes_uncertainty_v1/weights.pth"
model_config = f"contact_graspnet/gaze_model/inference/model/pretrained_weights/social_eyes_uncertainty_v1/config.yaml"
model_device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables to enable receiving the data stream, only for Linux.",
    )
    parser.add_argument(
        "--device-ip",
        default=None,
        type=str,
        help="Set glasses IP address for connection",
    )
    return parser.parse_args()

def gaze_inference(data: np.ndarray, inference_model, rgb_stream_label, device_calibration, rgb_camera_calibration):
    depth_m = 1  # 1 m

    # Iterate over the data and LOG data as we see fit
    img = torch.tensor(
        data, device="cuda"
    )

    with torch.no_grad():
        preds, lower, upper = inference_model.predict(img)
        preds = preds.detach().cpu().numpy()
        lower = lower.detach().cpu().numpy()
        upper = upper.detach().cpu().numpy()

    eye_gaze = EyeGaze
    eye_gaze.yaw = preds[0][0]
    eye_gaze.pitch = preds[0][1]

    # Compute eye_gaze vector at depth_m reprojection in the image
    gaze_projection = get_gaze_vector_reprojection(
        eye_gaze,
        rgb_stream_label,
        device_calibration,
        rgb_camera_calibration,
        depth_m,
    )

    # Adjust for image rotation
    width = 1408
    if gaze_projection.any() is None:
        return (0, 0)
    x, y = gaze_projection
    rotated_x = width - y
    rotated_y = x

    return (rotated_x, rotated_y)

def display_text(image, text: str, position, color=(0, 0, 255)):
    cv2.putText(
        img = image,
        text = text,
        org = position,
        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 1,
        color = color,
        thickness = 3
    )
import math

def rotation_matrix_to_euler(R):
    """
    Converts a rotation matrix to Euler angles (roll, pitch, yaw) in degrees.

    Args:
        R: 3x3 rotation matrix.

    Returns:
        A tuple: (roll, pitch, yaw) in degrees.
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0

    # Convert to degrees
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

def goto(robot, realsense_streamer, gaze, depth_frame, TCR, refine=False):
    # right
    # print(TCR)
    # TCR[0,3] += 25
    # TCR[1,3] += 25

    # print('waiting for depth data...')
    # for i in range(3):
    #     _, rgb_image, depth_frame, depth_img = realsense_streamer.capture_rgbd()

    waypoint_cam = 1000.0*np.array(realsense_streamer.deproject(gaze, depth_frame))
    print('waypoint_cam: ' , waypoint_cam)
    print('computing transformation...')
    waypoint_rob = transform(np.array(waypoint_cam).reshape(1,3), TCR)

    # Get waypoints in robot frame
    ee_pos_desired =np.array(waypoint_rob)[0]
    lift_pos = ee_pos_desired #+ np.array([0,0,100])
 
    lift_pos[2] = max(lift_pos[2], 250)
    lift_pos[2] = min(lift_pos[2], 290)
    # Put robot in canonical orientation
    robot.go_home()
    ee_pos, ee_euler = robot.pose_ee()

    state_log = robot.move_to_ee_pose(
        ee_pos, ee_euler, 
    )
    _, ee_euler = robot.pose_ee()
    # # #

    print('robot moving to', lift_pos, ee_euler)
    state_log = robot.move_to_ee_pose(
        lift_pos, ee_euler, 
    )

    # state_log = robot.move_to_ee_pose(
    #     ee_pos_desired, ee_euler, 
    # )

    # state_log = robot.move_to_ee_pose(
    #     lift_pos, ee_euler, 
    # )



def main():
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    # Initialize model for inference using path to model weight and config file
    model = infer.EyeGazeInference(
        model_weights,
        model_config,
        model_device
    )

    # Initialize Homography Manager
    homography_manager = HomographyManager(serial_no=serial_no)
    try:
        homography_manager.start_camera()
    except Exception as e:
        print(f"Error starting RealSense camera: {e}")
        # Handle the error appropriately, e.g., continue without homography if needed
        sys.exit("RealSense camera failed to start, exiting.") # Exit if RealSense camera fails to start

    # Initialize the Gaze History
    gaze_history = GazeHistory()

    # create aruco markers
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    parameters = cv2.aruco.DetectorParameters()

    #  Optional: Set SDK's log level to Trace or Debug for more verbose logs. Defaults to Info
    aria.set_log_level(aria.Level.Info)

    # 1. Create and connect the DeviceClient for fetching device calibration, which is required to cast 3D gaze prediction to 2D image
    device_client = aria.DeviceClient()
    client_config = aria.DeviceClientConfig()
    if args.device_ip:
        client_config.ip_v4_address = args.device_ip
    device_client.set_client_config(client_config)
    device = device_client.connect()

    # 2. Create StreamingClient instance and retrieve streaming_manager
    streaming_manager = device.streaming_manager
    streaming_client = aria.StreamingClient()

    # 3. Configure subscription to listen to Aria's RGB streams.
    # @see StreamingDataType for the other data types
    config = streaming_client.subscription_config
    config.subscriber_data_type = (
        aria.StreamingDataType.Rgb | aria.StreamingDataType.EyeTrack
    )

    # A shorter queue size may be useful if the processing callback is always slow and you wish to process more recent data
    # For visualizing the images, we only need the most recent frame so set the queue size to 1
    config.message_queue_size[aria.StreamingDataType.Rgb] = 1
    config.message_queue_size[aria.StreamingDataType.EyeTrack] = 1

    # Set the security options
    # @note we need to specify the use of ephemeral certs as this sample app assumes
    # aria-cli was started using the --use-ephemeral-certs flag
    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options
    streaming_client.subscription_config = config

    # 4. Create and attach observer to streaming client
    class StreamingClientObserver:
        def __init__(self):
            self.images = {}

        def on_image_received(self, image: np.array, record: ImageDataRecord):
            self.images[record.camera_id] = image

    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)

    # 5. Start listening
    print("Start listening to image data")
    streaming_client.subscribe()

    # 6. Visualize the streaming data until we close the window
    rgb_window = "Aria RGB"
    realsense_window = "RealSense ArUco Detection" # New window for RealSense

    cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rgb_window, 1024, 1024)
    cv2.setWindowProperty(rgb_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(rgb_window, 50, 50)

    cv2.namedWindow(realsense_window, cv2.WINDOW_NORMAL) # Create RealSense window
    cv2.resizeWindow(realsense_window, 640, 480) # Resize to RealSense resolution
    cv2.moveWindow(realsense_window, 1100, 50) # Position RealSense window

    # 7. Fetch calibration and labels to be passed to 3D -> 2D gaze coordinate casting function
    rgb_stream_label = "camera-rgb"
    device_calibration = streaming_manager.sensors_calibration()

    # sensors_calibration() returns device calibration in JSON format, so we must parse through in order to find the calibration for the RGB camera
    parser = json.loads(device_calibration)
    rgb_camera_calibration = next(
        camera for camera in parser['CameraCalibrations']
        if camera['Label'] == 'camera-rgb'
    )

    # Convert device calibration from JSON string to DeviceCalibration Object
    device_calibration = device_calibration_from_json_string(device_calibration)

    # Extract translation and quaternion variables from camera calibration JSON and preprocess
    translation = rgb_camera_calibration["T_Device_Camera"]["Translation"]
    quaternion = rgb_camera_calibration["T_Device_Camera"]["UnitQuaternion"]
    # quaternion format is [w, [x, y, z]]
    quat_w = quaternion[0]
    quat =[quaternion[1][0], quaternion[1][1], quaternion[1][2]]
    # convert both to numpy arrays with shape (3, 1)
    quat = np.array(quat).reshape(3, 1)
    translation = np.array(translation).reshape(3, 1)

    # Create SE3 Object containing information on quaternion coordinate and translations
    se3_transform = SE3.from_quat_and_translation(quat_w, quat, translation)

    # Retrieve RGB camera calibration from SE3 Object
    # The focal length can also be 611.1120152034575
    rgb_camera_calibration = get_linear_camera_calibration(1408, 1408, 550, 'camera-rgb', se3_transform) # the dimensions of the RGB camera is (1408, 1408)

    np.set_printoptions(threshold=np.inf) # set print limit to inf

    ariaCorners = None
    ariaIds = None
    gaze_coordinates = None
    last_gaze = None

    

    
    

    

    # 8. Continuously loop through and run gaze estimation + postprocessing on every frame before displaying
    while not quit_keypress():
        # Render the RGB image
        try:
            #print('looping')
            if aria.CameraId.Rgb in observer.images:
                rgb_image = np.rot90(observer.images[aria.CameraId.Rgb], -1)
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

                # Detect ArUco markers in Aria Image
                detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
                ariaCorners, ariaIds, rejected = detector.detectMarkers(gray)

                # Draw detected markers on Aria image
                if ariaIds is not None:
                    cv2.aruco.drawDetectedMarkers(rgb_image, ariaCorners, ariaIds)

                gaze = observer.images.get(aria.CameraId.EyeTrack)

                if gaze is not None and np.mean(gaze) > 10 and np.median(gaze) > 10:

                    # Run inference using gaze estimation model
                    gaze_coordinates = gaze_inference(gaze, model, rgb_stream_label, device_calibration, rgb_camera_calibration)

                    # If gaze coordinates exist, plot as a bright green dot on screen on Aria Image
                    if gaze_coordinates is not None:
                        cv2.circle(rgb_image, (int(gaze_coordinates[0]), int(gaze_coordinates[1])), 5, (0, 255, 0), 10)

                        # this code here can be used to test the gaze_history logging function
                        # this will log where the user is looking at, and return true if the user stares at an object for a set amount of time
                        # if gaze_history.log(gaze_coordinates):
                        #     print(f'USER IS STARING AT {gaze_history.last_gaze_stare}')

                    # Log coordinates of gaze with text on Aria Image
                    display_text(rgb_image, f'Gaze Coordinates: ({round(gaze_coordinates[0], 4)}, {round(gaze_coordinates[1], 4)})', (20, 90))

                else:
                    display_text(rgb_image, 'No Gaze Found', (20, 50))
                    gaze_coordinates = None

                cv2.imshow(rgb_window, rgb_image)
                del observer.images[aria.CameraId.Rgb]

                # Process RealSense frame and apply homography
                realsense_image, transformed_x, transformed_y = homography_manager.process_frame(
                    ariaCorners, ariaIds, gaze_coordinates
                )

                if realsense_image is not None:
                    # Display the RealSense image (e.g., in a new window) and plot on cv2 image
                    if transformed_x is not None and transformed_y is not None and gaze_coordinates is not None:
                        cv2.circle(realsense_image, (int(transformed_x), int(transformed_y)), 5, (0, 255, 0), 10)
                        cv2.putText(realsense_image, f'Transformed Gaze: ({round(transformed_x, 2)}, {round(transformed_y, 2)})', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        point_3d = homography_manager.get_vector(transformed_x, transformed_y)
                        if point_3d is not None:

                            # If you want to visualize it in 2D, you need to project the 3D point to 2D (for rendering on the image)
                            projected_point = homography_manager.projected_point_test(point_3d)
                            
                            # Draw a line from (0,0) to the projected gaze point on the 2D image
                            start = (int(0), int(0))  # Camera center (for 2D)
                            end = (int(transformed_x), int(transformed_y))  # Projected gaze point
                            
                            cv2.line(realsense_image, start, end, (0, 0, 255), 2)  # Draw a red line

                    
                        # gaze history returns true if user has been staring at the same (x, y) coordinate point for x amount of frames
                        if gaze_history.log((transformed_x, transformed_y)):
                            print(f'USER IS STARING AT {gaze_history.report_stare_coordinates()}')
                            # gemini_model.inference3D(realsense_image)
                            # gemini_model.inference2D(realsense_image)
                        
                            gaze_median = gaze_history.history_median()
                            if not gaze_median is None:
                                gaze = [int(np.floor(gaze_median[0])), int(np.floor(gaze_median[1]))]                        
                            else:
                                gaze = [int(np.floor(transformed_x)), int(np.floor(transformed_y))]    
                            print("gaze: ", gaze)
                            #goto(robot, homography_manager.realsense_streamer, gaze, homography_manager.depth_frame, TCR, refine=True)

                        # res = input('save?')
                        # if res == 'y' or res == 'Y':
                        #     np.save('calib/transforms.npy', transforms)
                        #     print('SAVED')

                    else:
                        cv2.putText(realsense_image, 'No Transformation', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    cv2.imshow(realsense_window, realsense_image)
                    ''' # TRYING TO FIGURE OUT THE GRASP TO THE ROBOT TRANSFORMATION
                    closest_grasp = np.array([[ 0.49967167,  0.8235121  ,-0.2686188,   0.00603566],
                            [-0.1900385 ,  0.40676996 , 0.8935455  ,-0.14829922],
                            [ 0.84511155 ,-0.39543146 , 0.35975048 , 0.6978372 ],
                            [ 0.          ,0.         , 0.         , 1.        ]])
                    position_cam = 1000.0*np.array(closest_grasp[:3, 3])  # Extract translation
                    position_rob = np.array(transform(np.array(position_cam).reshape(1,3), TCR))

                    # Extract rotation matrix and convert to Euler angles (roll, pitch, yaw)
                    rotation_matrix = closest_grasp[:3, :3]
                    roll, pitch, yaw = rotation_matrix_to_euler(rotation_matrix)

                    orientation = [roll, pitch, yaw]
                    print("Position: ", position_rob)
                    print("Orientation: ", orientation)
                    robot.move_to_ee_pose(position_rob[0], [1.79132906e+02 ,-1.00840000e-02 , 7.75670000e-01])
                    time.sleep(20)'''


        except Exception as e:
            print(f'Encountered error: {e}')

    # 9. Unsubscribe to clean up resources
    print("Stop listening to image data")
    streaming_client.unsubscribe()

    

if __name__ == "__main__":
    main()