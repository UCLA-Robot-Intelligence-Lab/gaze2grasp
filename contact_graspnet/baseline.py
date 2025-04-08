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

from homography import HomographyManager
from gaze_history import GazeHistory

from multicam import XarmEnv

from scipy.spatial.transform import Rotation as R
from rs_streamer import RealsenseStreamer
from calib_utils.linalg_utils import transform
from segment.FastSAM.fastsam import FastSAM
from segment.SAMInference import select_from_sam_everything




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
    '''
    Move the robot to the gaze point in the camera frame
    '''
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

    print('robot moving to', lift_pos, ee_euler)
    state_log = robot.move_to_ee_pose(
        lift_pos, ee_euler, 
    )


if __name__ == "__main__":

    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    # ----------------------------
    # set up the realsense camera
    # ----------------------------
    serial_no = '317422074281'
    realsense_streamer = RealsenseStreamer(serial_no)

    transforms = np.load(f'calib/transforms.npy', allow_pickle=True).item()
    TCR = transforms[serial_no]['tcr']

    homography_manager = HomographyManager(serial_no=serial_no)
    try:
        homography_manager.start_camera()
    except Exception as e:
        print(f"Error starting RealSense camera: {e}")
        # Handle the error appropriately, e.g., continue without homography if needed
        sys.exit("RealSense camera failed to start, exiting.") # Exit if RealSense camera fails to start

    # ----------------------------
    # set up the gaze inferenece
    # ----------------------------
    model_weights = f"contact_graspnet/gaze_model/inference/model/pretrained_weights/social_eyes_uncertainty_v1/weights.pth"
    model_config = f"contact_graspnet/gaze_model/inference/model/pretrained_weights/social_eyes_uncertainty_v1/config.yaml"
    model_device = "cuda" if torch.cuda.is_available() else "cpu"

    model = infer.EyeGazeInference(
        model_weights,
        model_config,
        model_device
    )

    gaze_history = GazeHistory()

    # ----------------------------
    # set up the robot
    # ----------------------------
    robot = XarmEnv()

    # ----------------------------
    # set up the aria
    # ----------------------------
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

    config.message_queue_size[aria.StreamingDataType.Rgb] = 1
    config.message_queue_size[aria.StreamingDataType.EyeTrack] = 1

    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options
    streaming_client.subscription_config = config

    class StreamingClientObserver:
        def __init__(self):
            self.images = {}

        def on_image_received(self, image: np.array, record: ImageDataRecord):
            self.images[record.camera_id] = image

    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)

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

    # ----------------------------
    # set up the aruco markers
    # ----------------------------
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    aruco_movement_map = {
        0: "pos_x_pos",
        1: "pos_x_neg",
        2: "pos_y_pos",
        3: "pos_y_neg",
        4: "pos_z_pos",
        5: "pos_z_neg",
        6: "rot_x_pos",
        7: "rot_x_neg",
        8: "rot_y_pos",
        9: "rot_y_neg",
        10: "rot_z_pos",
        11: "rot_z_neg",
        12: "grasp"
    }
    
    ariaCorners = None
    ariaIds = None
    gaze_coordinates = None
    last_gaze = None

    seg_model = FastSAM('./contact_graspnet/segment/FastSAM-s.pt')
    
    while not quit_keypress():
        try:
            if aria.CameraId.Rgb in observer.images:

                # detect which aruco markers are gazed at
                gaze = observer.images.get(aria.CameraId.EyeTrack)

                if gaze is not None and np.mean(gaze) > 10 and np.median(gaze) > 10:

                    # Run inference using gaze estimation model
                    gaze_coordinates = gaze_inference(gaze, model, rgb_stream_label, device_calibration, rgb_camera_calibration)

                    # If gaze coordinates exist, plot as a bright green dot on screen on Aria Image
                    if gaze_coordinates is not None:
                        cv2.circle(rgb_image, (int(gaze_coordinates[0]), int(gaze_coordinates[1])), 5, (0, 255, 0), 10)

                cv2.imshow(rgb_window, rgb_image)
                del observer.images[aria.CameraId.Rgb]

                # segment that aruco marker from the image
                segmented_cam_img, _ = select_from_sam_everything( #Segments everything and merge masks that is closest to the point prompt
                                    seg_model,
                                    [gaze],
                                    input_img=realsense_image,
                                    imgsz=640,
                                    iou=0.9,
                                    conf=0.4,
                                    max_distance=10,#100,
                                    device=None,
                                    retina=True,
                                    #include_largest_mask = True
                                )
                
                mask = segmented_cam_img.astype(np.uint8)
                if np.max(mask) <= 1:
                    mask *= 255

                masked_image = cv2.bitwise_and(realsense_image, realsense_image, mask=mask)
                gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

                # get the aruco marker's id
                corners, ids, _ = detector.detectMarkers(gray)

                if ids is not None:
                    print("Detected ArUco ID(s):", ids.flatten().tolist())
                else:
                    print("No ArUco markers detected.")

                # map the marker's id to the movement of the robot arm
                if ids is not None:
                    print("Detected ArUco ID(s):", ids.flatten().tolist())
                    for marker_id in ids.flatten():
                        action = aruco_movement_map.get(marker_id, "unknown")
                        print(f"Marker ID {marker_id}: Action → {action}")
                    
                # move the arm



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


                # Process RealSense frame and apply homography
                realsense_image, transformed_x, transformed_y = homography_manager.process_frame(
                    ariaCorners, ariaIds, gaze_coordinates
                )

                # if realsense_image is not None:
                #     # Display the RealSense image (e.g., in a new window) and plot on cv2 image
                #     if transformed_x is not None and transformed_y is not None and gaze_coordinates is not None:
                #         cv2.circle(realsense_image, (int(transformed_x), int(transformed_y)), 5, (0, 255, 0), 10)
                #         cv2.putText(realsense_image, f'Transformed Gaze: ({round(transformed_x, 2)}, {round(transformed_y, 2)})', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                #         # gaze history returns true if user has been staring at the same (x, y) coordinate point for x amount of frames
                #         if gaze_history.log((transformed_x, transformed_y)):
                #             print(f'USER IS STARING AT {gaze_history.report_stare_coordinates()}')

                #             gaze_median = gaze_history.history_median()
                #             if not gaze_median is None:
                #                 gaze = [int(np.floor(gaze_median[0])), int(np.floor(gaze_median[1]))]                        
                #             else:
                #                 gaze = [int(np.floor(transformed_x)), int(np.floor(transformed_y))]    
                #             print("gaze: ", gaze)
                #             goto(robot, homography_manager.realsense_streamer, gaze, homography_manager.depth_frame, TCR, refine=True)

                #     else:
                #         cv2.putText(realsense_image, 'No Transformation', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                #     cv2.imshow(realsense_window, realsense_image)
        except Exception as e:
            print(f'Encountered error: {e}')


# control the robot arm correspondingly
