import argparse
import cv2
import json
import numpy as np
import sys
import torch
import os
import aria.sdk as aria
import time
from segment.FastSAM.fastsam import FastSAM
from segment.SAMInference import select_from_sam_everything, run_fastsam_point_inference
from grasp_selector import find_closest_grasp

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from contact_grasp_estimator import GraspEstimator

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
from calib.rotation_transform import transform_rotation_camera_to_robot_roll_yaw_pitch
from calib_utils.linalg_utils import transform
import config_utils

import open3d as o3d


GRIPPER_SPEED, GRIPPER_FORCE, GRIPPER_MAX_WIDTH, GRIPPER_TOLERANCE = 0.1, 40, 0.08570, 0.01

serial_no = '317422075456'

transforms = np.load(f'calib/transforms_{serial_no}.npy', allow_pickle=True).item()
TCR = transforms[serial_no]['tcr']

robot = XarmEnv()

seg_model = FastSAM('./contact_graspnet/segment/FastSAM-s.pt')



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




def main():
    # file paths to model weights and configuration
    model_weights = f"contact_graspnet/gaze_model/inference/model/pretrained_weights/social_eyes_uncertainty_v1/weights.pth"
    model_config = f"contact_graspnet/gaze_model/inference/model/pretrained_weights/social_eyes_uncertainty_v1/config.yaml"
    model_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='checkpoints/scene_test_2048_bs3_hor_sigma_001', help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]')
    parser.add_argument('--np_path', default='test_data/0.npy', help='Input data path')
    parser.add_argument('--K', default=5, type=int, help='Number of grasps to generate [default: 5]')
    parser.add_argument('--num_samples', default=1024, type=int, help='Number of samples to evaluate during testing, and the number of points in each sample')
    parser.add_argument('--show_non_collision', default=False, action='store_true')

    parser.add_argument(
            "--model_weights",
            default=model_weights,
            help="The path to the model weights file. Could either be a local file or on the server",
            type=str,
        )
    parser.add_argument(
        "--model_config",
        default=model_config,
        help="The path to the model config file. Could either be a local file or on the server.",
        type=str,
    )

    parser.add_argument(
        "--device_ip",
        default="",
        help="Device IPv4 address, retrieved from `DeviceClientConfigProvider` if not provided",
        type=str,
    )

    parser.add_argument(
        "--serial_no",
        default="",
        help="Device Serial Number, retrieved from `DeviceClientConfigProvider` if not provided",
        type=str,
    )

    parser.add_argument(
        "--model_device",
        default=model_device,
        help="Device used to run inference, chosen between 'cpu' and 'cuda'. Defaults to 'cpu'",
        type=str,
    )
    parser.add_argument(
        "--update_iptables",
        default=True,
        help="""
        Update IP table with local device's IP. Defaults to True.
        Should only be enabled on linux
        """,
        type=bool,
    )


    parser.add_argument('--z_range', default=[0.2,1.8], help='Z value threshold to crop the input point cloud')
    parser.add_argument('--local_regions', action='store_true', default=False, help='Crop 3D local regions around given segments.')
    parser.add_argument('--filter_grasps', action='store_true', default=False,  help='Filter grasp contacts according to segmap.')
    parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
    parser.add_argument('--forward_passes', type=int, default=1,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    parser.add_argument('--segmap_id', type=int, default=0,  help='Only return grasps of the given object id')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    FLAGS = parser.parse_args()

    global_config = config_utils.load_config(FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs)

    print(str(global_config))
    print('pid: %s'%(str(os.getpid())))

    # Instantiate GraspEstimator - this line causes the error.
    grasp_estimator = GraspEstimator(global_config) 

    grasp_estimator.build_network()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(save_relative_paths=True)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    print('Session created: ', sess.list_devices())

    # Load weights
    grasp_estimator.load_weights(sess, saver, 'checkpoints/scene_test_2048_bs3_hor_sigma_001', mode='test')
    os.makedirs('results', exist_ok=True)

    args = parser.parse_args()

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
    seg_window = "Gaze Segmentation (Aria)" # New window for RealSense
    seg_cam_window = "Gaze Segmentation (Camera)" # New window for RealSense


    cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rgb_window, 1024, 1024)
    cv2.setWindowProperty(rgb_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(rgb_window, 50, 50)

    cv2.namedWindow(realsense_window, cv2.WINDOW_NORMAL) # Create RealSense window
    cv2.resizeWindow(realsense_window, 640, 480) # Resize to RealSense resolution
    cv2.moveWindow(realsense_window, 1100, 50) # Position RealSense window

    cv2.namedWindow(seg_window, cv2.WINDOW_NORMAL) 
    cv2.resizeWindow(seg_window, 640, 480) 
    cv2.moveWindow(seg_window, 1800, 50) 

    cv2.namedWindow(seg_cam_window, cv2.WINDOW_NORMAL) 
    cv2.resizeWindow(seg_cam_window, 640, 480) 
    cv2.moveWindow(seg_cam_window, 1800, 1050) 


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

    # Extract translation and quarternon variables from camera calibration JSON and preprocess
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
                    rgb_image_raw = rgb_image.copy()
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
                        
                        # gaze history returns true if user has been staring at the same (x, y) coordinate point for x amount of frames
                        if gaze_history.log((transformed_x, transformed_y)):
                            print(f'USER IS STARING AT {gaze_history.report_stare_coordinates()}')

                            gaze_median = gaze_history.history_median()
                            if not gaze_median is None:
                                gaze = ([int(np.floor(gaze_median[0])), int(np.floor(gaze_median[1]))]  )                      
                            else:
                                gaze = ([int(np.floor(transformed_x)), int(np.floor(transformed_y))]    )
                            #print("gaze_coordinates_aria: ", gaze_coordinates)
                            print('rgb_image shape:', (rgb_image.shape))
                            try:

                                segmented_img, mask_points = select_from_sam_everything( #Segments everything and merge masks that is closest to the point prompt
                                                    seg_model,
                                                    gaze_coordinates,
                                                    input_img=rgb_image_raw,
                                                    imgsz=1408,
                                                    iou=0.9,
                                                    conf=0.4,
                                                    max_distance=50,#100,
                                                    device=None,
                                                    retina=True,
                                                )
                            
                                print("segmented_img shape:", segmented_img.shape)
                                print("segmented_img dtype:", segmented_img.dtype)
                                print("segmented_img min/max:", segmented_img.min(), segmented_img.max())
                                
                                cv2.imshow(seg_window, segmented_img)
                                print("mask_points: ", mask_points)
                                time.sleep(0.5)
                            except Exception as e:
                                print(f"Error displaying segmented image: {e}")
                                break

                            mask_pt_cam = []
                            for mask_point in mask_points:
                                transformed_x, transformed_y = homography_manager.apply_homography(mask_point)
                                if transformed_x < 0 or transformed_y < 0 or transformed_x >= 640 or transformed_y >= 480:
                                    print("Point out of bounds")
                                    break
                                mask_pt_cam.append([int(np.floor(transformed_x)), int(np.floor(transformed_y))])
                            print("mask_pt_cam: ", mask_pt_cam)
                            try:
                                segmented_cam_img = run_fastsam_point_inference(
                                    seg_model,
                                    mask_pt_cam,
                                    input_img=realsense_image,
                                    imgsz=640,
                                    iou=0.9,
                                    conf=0.4,
                                    point_label="[1,0]",
                                    device=None,
                                    retina=True,
                                )
                                
                                print("segmented_cam_img shape:", segmented_cam_img.shape)
                                print("segmented_cam_img dtype:", segmented_cam_img.dtype)
                                print("segmented_cam_img min/max:", segmented_cam_img.min(), segmented_cam_img.max())
                                cv2.imshow(seg_cam_window, segmented_cam_img.astype(np.uint8) * 255)
                                time.sleep(0.5)

                            except Exception as e:
                                print(f"Error displaying segmented image: {e}")
                                break
                            
                            pc_full, pc_color = homography_manager.realsense_streamer.seg_to_pc(segmented_cam_img)

                            # Create an Open3D point cloud object
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(pc_full)
                            pcd.colors = o3d.utility.Vector3dVector(pc_color)

                            # Visualize the point cloud
                            o3d.visualization.draw_geometries([pcd])

                            grasps, _, _, _ = grasp_estimator.predict_scene_grasps(sess, pc_full, pc_segments=None, 
                                                                                          local_regions=None, filter_grasps=False, forward_passes=1)  
                            arrows = []
                            grasp_width = 0.008

                            # Find the closest grasp to the gaze point using nearest neighbour
                            #closest_grasp = find_closest_grasp(grasps, gaze, homography_manager.realsense_streamer.depth_frame, homography_manager.realsense_streamer)

                            #Plot only the centre grasp
                            center = pcd.get_center()

                            # Calculate distances from grasp translations to the center
                            distances = [np.linalg.norm(grasp[:3, 3] - center) for grasp in grasps]

                            # Find the index of the closest grasp
                            closest_grasp_index = np.argmin(distances)
                            closest_grasp = grasps[closest_grasp_index]
                            arrows = []

                            # Extract rotation matrix (3x3) and translation (position)
                            #R = np.identity(3)
                            R = np.array(closest_grasp[:3, :3])
                            t = np.array(closest_grasp[:3, 3])

                            grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=t)  # Adjust size as needed
                            grasp_frame.rotate(R, center=t) # Apply the grasp rotation


                            # Define the grasp approach vector (z-axis in local gripper frame)
                            approach_vector = np.array([0, 0, 0.1])  # Arrow length of 10cm

                            # Transform the approach vector to world coordinates
                            v_transformed = R @ approach_vector + t

                            # Define grasp width vector for visualization (left and right finger placement)
                            grasp_offset = R @ np.array([grasp_width / 2, 0, 0])  # Half of width along x-axis
                            left_finger = t - grasp_offset
                            right_finger = t + grasp_offset

                            # Create arrow for approach direction
                            approach_arrow = o3d.geometry.LineSet()
                            approach_arrow.points = o3d.utility.Vector3dVector([t, v_transformed])
                            approach_arrow.lines = o3d.utility.Vector2iVector([[0, 1]])
                            approach_arrow.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # Green (Approach)

                            # Create line for grasp width (finger placement)
                            grasp_line = o3d.geometry.LineSet()
                            grasp_line.points = o3d.utility.Vector3dVector([left_finger, right_finger])
                            grasp_line.lines = o3d.utility.Vector2iVector([[0, 1]])
                            grasp_line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red (Grasp width)

                            arrows.append(approach_arrow)
                            arrows.append(grasp_line)
                            arrows.append(grasp_frame)
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(pc_full)
                            pcd.colors = o3d.utility.Vector3dVector(pc_color)

                            print("Closest predicted grasp: ", closest_grasp)
                            position_cam = 1000.0*np.array(closest_grasp[:3, 3])  # Extract translation
                            position_rob = np.array(transform(np.array(position_cam).reshape(1,3), TCR))[0]

                            # Extract rotation matrix and convert to Euler angles (roll, pitch, yaw)
                            rotation_matrix = closest_grasp[:3, :3]
                            
                            # Visualize with point cloud
                            o3d.visualization.draw_geometries([pcd] + arrows)
                            orientation = transform_rotation_camera_to_robot_roll_yaw_pitch(rotation_matrix, TCR)
                            print("Position: ", position_rob)
                            print("Orientation: ", orientation)
                            robot.move_to_ee_pose(position_rob, orientation)

        

                    else:
                        cv2.putText(realsense_image, 'No Transformation', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    cv2.imshow(realsense_window, realsense_image)


        except Exception as e:
            print(f'Encountered error: {e}')

    # 9. Unsubscribe to clean up resources
    print("Stop listening to image data")
    streaming_client.unsubscribe()

    

if __name__ == "__main__":
    main()