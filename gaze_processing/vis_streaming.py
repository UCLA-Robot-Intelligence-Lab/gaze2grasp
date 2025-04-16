import sys
import argparse
import time
import cv2
import numpy as np
from aria_glasses import AriaGlasses
from aria_glasses.utils.general import quit_keypress, update_iptables, read_vis_params, read_gaze_vis_params, display_text
from aria_glasses.utils.streaming import visualize_streaming
from aria_glasses.utils.config_manager import ConfigManager
import pdb


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
        default="./gaze_processing/my_aria_config.yaml",
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

window_name, window_size, window_position = read_vis_params(config_manager)
rgb_window = window_name
cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(rgb_window, window_size[0], window_size[1])
cv2.setWindowProperty(rgb_window, cv2.WND_PROP_TOPMOST, 1)
cv2.moveWindow(rgb_window, window_position[0], window_position[1])
gaze_vis_params = read_gaze_vis_params(config_manager)

while not quit_keypress():
    gaze = glasses.infer_gaze(mode='2d')
    rgb_image = glasses.get_frame_image('rgb')

    if rgb_image is not None:
        visualize_streaming(rgb_window, rgb_image, gaze_vis_params, gaze)

glasses.stop_streaming()
