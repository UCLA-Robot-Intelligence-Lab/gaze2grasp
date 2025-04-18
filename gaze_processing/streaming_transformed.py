import sys
import argparse
import time
import cv2
import numpy as np
from aria_glasses import AriaGlasses
from aria_glasses.utils.general import quit_keypress, update_iptables, read_vis_params, read_gaze_vis_params, display_text
from aria_glasses.utils.streaming import visualize_streaming
from aria_glasses.utils.config_manager import ConfigManager
from homography import HomographyManager, SERIAL_NO_81, SERIAL_NO_56
from gaze_history import GazeHistory


#===========================SETTING UP ARIA STREAMING===========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables to enable receiving the data stream, only for Linux.",
    )
    
    parser.add_argument(
        "--device_ip",
        default=None,
        type=str,
        help="Set glasses IP address for connection",
    )

    parser.add_argument(
        "--config_path",
        default="baseline/my_aria_config.yaml",
        action="store_true",
        help="Specify the path to the configuration file",
    )
    return parser.parse_args()

args = parse_args()
if args.update_iptables and sys.platform.startswith("linux"):
    update_iptables()
device_ip = args.device_ip
config_path = args.config_path
config_manager = ConfigManager(config_path)

# Initialize AriaGlasses
glasses = AriaGlasses(device_ip, config_path)

glasses.start_streaming()

#=============SETTING UP HOMOGRAPHY, REALSENSE CAMERAS AND GAZE HISTORY============

homography_manager_81 = HomographyManager(serial_no=SERIAL_NO_81)
homography_manager_56 = HomographyManager(serial_no=SERIAL_NO_56)
homography_manager = [homography_manager_81, homography_manager_56]

try:
    for i in range(2):
        homography_manager[i].start_camera()
except Exception as e:
    print(f"Error starting RealSense camera: {e}")
    sys.exit("RealSense camera failed to start, exiting.") 

# create aruco markers
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
parameters = cv2.aruco.DetectorParameters()

ariaCorners = None
ariaIds = None
gaze_coordinates = None
last_gaze = None

# Initialize the Gaze History
gaze_history = GazeHistory()

#===========================VISUALIZATION===========================
# Set up visualization for ARIA glasses
window_name, window_size, window_position = read_vis_params(config_manager)
rgb_window = window_name
cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(rgb_window, window_size[0], window_size[1])
cv2.setWindowProperty(rgb_window, cv2.WND_PROP_TOPMOST, 1)
cv2.moveWindow(rgb_window, window_position[0], window_position[1])
gaze_vis_params = read_gaze_vis_params(config_manager)

# Set up visualization for RealSense cameras
realsense_window_81 = f"RealSense ArUco Detection ({SERIAL_NO_81})" 
cv2.namedWindow(realsense_window_81, cv2.WINDOW_NORMAL) 
cv2.resizeWindow(realsense_window_81, 640, 480)
cv2.moveWindow(realsense_window_81, 1100, 50)

realsense_window_56 = f"RealSense ArUco Detection ({SERIAL_NO_56})"
cv2.namedWindow(realsense_window_56, cv2.WINDOW_NORMAL) 
cv2.namedWindow(realsense_window_56, cv2.WINDOW_NORMAL) 
cv2.resizeWindow(realsense_window_56, 640, 480) 
cv2.moveWindow(realsense_window_56, 1800, 50) 
#===========================


while not quit_keypress():
    gaze = glasses.infer_gaze(mode='2d')
    rgb_image = glasses.get_frame_image('rgb')
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    # Detect ArUco markers in Aria Image
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    ariaCorners, ariaIds, rejected = detector.detectMarkers(gray)

    realsense_image_81, x_81, y_81 = homography_manager_81.process_frame(ariaCorners, ariaIds, gaze)
    realsense_image_56, x_56, y_56 = homography_manager_56.process_frame(ariaCorners, ariaIds, gaze)

    if rgb_image is not None:
        visualize_streaming(rgb_window, rgb_image, gaze_vis_params, gaze)
    if realsense_image_81 is not None and np.array([x_81, y_81]).all() is not None:
        visualize_streaming(realsense_window_81, realsense_image_81, gaze_vis_params, [x_81, y_81])
    if realsense_image_56 is not None and np.array([x_56, y_56]).all() is not None:
        visualize_streaming(realsense_window_56, realsense_image_56, gaze_vis_params, [x_56, y_56])

glasses.stop_streaming()
