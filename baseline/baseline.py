import sys
import argparse
import time
import cv2
import cv2.aruco as aruco
import numpy as np
from aria_glasses import AriaGlasses
from aria_glasses.utils.general import quit_keypress, update_iptables, read_vis_params, read_gaze_vis_params, display_text
from aria_glasses.utils.streaming import visualize_streaming
from aria_glasses.utils.config_manager import ConfigManager
import pdb
from xarm.wrapper import XArmAPI

# initialize the robot
arm = XArmAPI("192.168.1.223")

def go_home():
    arm.motion_enable(enable=True)  # change to False under Sim!!!!!

    arm.set_mode(0)
    arm.set_state(state=0)
    arm.set_servo_angle(angle=[0, 0, 0, 105, 0, 105, 0], speed=50, wait=True)

def expand_quad(corners, scale=1.05):
    '''
    expand the quad by 5% to ensure edge points are included
    '''
    center = np.mean(corners, axis=0)
    return center + (corners - center) * scale

def get_aruco_id(image, gaze, detector):
    gazed_id = None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    corners, ids, _ = detector.detectMarkers(thresholded)

    if ids is not None:
        for i, marker_corners in enumerate(corners):
            marker_corners = marker_corners.reshape((4, 2))     # reshape corners to 4x2 array (in order: TL, TR, BR, BL)
            
            marker_corners = expand_quad(marker_corners)
            if cv2.pointPolygonTest(marker_corners, gaze, False) >= 0:
                gazed_id = ids[i][0]
                print(f"The point is on ArUco marker ID: {gazed_id}")
                break

    return gazed_id

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
        default="./baseline/my_aria_config.yaml",
        action="store_true",
        help="Specify the path to the configuration file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()
    device_ip = args.device_ip
    config_path = args.config_path
    config_manager = ConfigManager(config_path)

    # initialize AriaGlasses
    glasses = AriaGlasses(device_ip, config_path)

    glasses.start_streaming()
    glasses.start_recording('./recordings')

    window_name, window_size, window_position = read_vis_params(config_manager)
    rgb_window = window_name
    cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rgb_window, window_size[0], window_size[1])
    cv2.setWindowProperty(rgb_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(rgb_window, window_position[0], window_position[1])
    gaze_vis_params = read_gaze_vis_params(config_manager)

    # initialize ArUco marker detection
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    aruco_params = aruco.DetectorParameters()
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # or CORNER_REFINE_APRILTAG
    detector = aruco.ArucoDetector(aruco_dict, aruco_params)

    # arm control
    # go_home()

    arm.motion_enable(enable=True)  # change to False under Sim!!!!!

    arm.set_mode(5)
    arm.set_state(state=0)
    time.sleep(1)
    arm.set_gripper_enable(True)
    v_pos = 60
    v_rot = 15
    gripper_state = False

    id_to_control = {
        0: [v_pos, 0, 0, 0, 0, 0],
        1: [0, v_pos, 0, 0, 0, 0],
        2: [-v_pos, 0, 0, 0, 0, 0],
        3: [0, -v_pos, 0, 0, 0, 0],
        4: [0, 0, v_pos, 0, 0, 0],
        5: [0, 0, -v_pos, 0, 0, 0],
        6: [0, 0, 0, v_rot, 0, 0],
        7: [0, 0, 0, 0, v_rot, 0],
        8: [0, 0, 0, -v_rot, 0, 0],
        9: [0, 0, 0, 0, -v_rot, 0],
        10: [0, 0, 0, 0, 0, v_rot],
        11: [0, 0, 0, 0, 0, -v_rot],
        12: gripper_state,
    }


    while not quit_keypress():
        # get gaze
        gaze = glasses.infer_gaze(mode='2d')
        rgb_image = glasses.get_frame_image('rgb')

        if rgb_image is not None:
            visualize_streaming(rgb_window, rgb_image, gaze_vis_params, gaze)

        # get which aruco marker is being looked at
        if rgb_image is not None:
            gazed_id = get_aruco_id(rgb_image, gaze, detector)

            # robot move correspondingly
            if gazed_id in id_to_control:
                if gazed_id == 12:
                    gripper_state = not gripper_state
                    arm.set_gripper_enable(gripper_state)
                else:
                    arm.vc_set_cartesian_velocity(id_to_control[gazed_id])
                    # time.sleep(0.01)
                    # arm.vc_set_cartesian_velocity([0, 0, 0, 0, 0, 0])
            else:
                arm.vc_set_cartesian_velocity([0, 0, 0, 0, 0, 0])

    glasses.stop_recording()
    glasses.stop_streaming()
    arm.close()

if __name__ == "__main__":
    main()

# TODO:
# 1. record the time people look at the markers for control and the time they look at the scene
