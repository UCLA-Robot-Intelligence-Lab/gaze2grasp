import pyrealsense2 as rs
import cv2
import os
import sys
import time
import open3d as o3d
import numpy as np

'''current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)'''

from load_np_file import load_np, full_path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))
from visualizations.live_visualization import segment_image, capture_and_process_rgbd
from scipy.spatial.transform import Rotation
from multicam import XarmEnv
from rs_streamer import RealsenseStreamer

# Constants
GRIPPER_SPEED, GRIPPER_FORCE, GRIPPER_MAX_WIDTH, GRIPPER_TOLERANCE = 0.1, 40, 0.08570, 0.01
SERIAL_NO_81 = '317422074281'  # Camera serial number
SERIAL_NO_56 = '317422075456'
# Load calibration data 
transforms = np.load(full_path('transforms.npy'), allow_pickle=True).item()
TCR_81 = transforms[SERIAL_NO_81]['tcr']
TCR_56 = transforms[SERIAL_NO_56]['tcr']
TCR_81[:3, 3] /= 1000.0
TCR_56[:3, 3] /= 1000.0

base_link_color = [[1, 0.6, 0.8],  [1, 1, 0], [1, 0.5, 0], [0.4, 0, 0.8]]
# Initialize robot
robot = XarmEnv()

class PixelSelector:
    def __init__(self):
        self.color_index = 0
        pass

    def load_image(self, img, recrop=False):
        self.img = img
        if recrop:
            cropped_img = self.crop_at_point(img, 700, 300, width=400, height=300)
            self.img = cv2.resize(cropped_img, (640, 480))

    def crop_at_point(self, img, x, y, width=640, height=480):
        img = img[y:y+height, x:x+width]
        return img

    def mouse_callback(self, event, x, y, flags, param):
        cv2.imshow("pixel_selector", self.img)
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.clicks.append([x, y])
            color_float = base_link_color[self.color_index % len(base_link_color)]
            color_to_use = tuple(int(c * 255) for c in reversed(color_float)) # Reversed for BGR
            cv2.circle(self.img, (x, y), 3, color_to_use, -1)
            self.color_index += 1

    def run(self, img):
        self.load_image(img)
        self.clicks = []
        cv2.namedWindow('pixel_selector')
        cv2.setMouseCallback('pixel_selector', self.mouse_callback)
        while True:
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
        return self.clicks, self.img

def calculate_final_pose(place_ee_world, averaged_pick_world, initial_pitch_deg=-50, final_pitch_deg=-10):
    """
    Calculates the final EE pose (position and orientation as roll, pitch, yaw)
    for a pick and place operation with a change in pitch while aligning
    with the approach vector.

    Args:
        place_ee_world (np.ndarray): Coordinates of the place end-effector in the world frame (3x1).
        averaged_pick_world (np.ndarray): Coordinates of the averaged pick point in the world frame (3x1).
        initial_pitch_deg (float): The initial fixed pitch angle in degrees.
        final_pitch_deg (float): The desired final pitch angle in degrees.

    Returns:
        tuple: A tuple containing:
            - pick_position (np.ndarray): The calculated pick position (3x1).
            - initial_rpy (list): Initial roll, pitch, yaw angles in degrees [roll, pitch, yaw].
            - place_position (np.ndarray): The calculated place position (3x1).
            - final_rpy (list): Final roll, pitch, yaw angles in degrees [roll, pitch, yaw].
    """
    approach_vector = place_ee_world - averaged_pick_world
    approach_vector_unit = approach_vector / np.linalg.norm(approach_vector)
    initial_z_axis = np.array([0, 0, 1])

    initial_pitch_rad = np.deg2rad(initial_pitch_deg)
    Ry_initial_pitch = Rotation.from_euler('y', initial_pitch_rad, degrees=False)
    pitched_z_axis = Ry_initial_pitch.apply(initial_z_axis)

    yaw_rad = np.arctan2(approach_vector_unit[1], approach_vector_unit[0])
    yaw_deg = np.rad2deg(yaw_rad)
    Rz_yaw = Rotation.from_euler('z', yaw_rad, degrees=False)
    yawed_pitched_z_axis = Rz_yaw.apply(pitched_z_axis)

    # Calculate the roll needed to align the x-y components
    angle_xy = np.arctan2(approach_vector_unit[1], approach_vector_unit[0]) - np.arctan2(yawed_pitched_z_axis[1], yawed_pitched_z_axis[0])
    roll_rad = angle_xy
    roll_deg = np.rad2deg(roll_rad)
    Rx_roll = Rotation.from_euler('x', roll_rad, degrees=False)

    initial_rotation = Rz_yaw * Ry_initial_pitch * Rx_roll  # Order matters: Roll -> Pitch -> Yaw (intrinsic)

    # Calculate pick and approach positions
    pick_position = averaged_pick_world + np.array([0, 0, 70])

    # Calculate final orientation
    rotation_axis = np.cross(initial_z_axis, approach_vector_unit)
    if np.linalg.norm(rotation_axis) < 1e-6:
        current_euler = initial_rotation.as_euler('xyz', degrees=False)
        final_pitch_rad = np.deg2rad(final_pitch_deg)
        final_rotation = Rotation.from_euler('xyz', [current_euler[0], final_pitch_rad, current_euler[2]], degrees=False)
    else:
        rotation_axis_unit = rotation_axis / np.linalg.norm(rotation_axis)
        delta_pitch_rad = np.deg2rad(final_pitch_deg) - initial_pitch_rad
        pitch_change_rotation = Rotation.from_rotvec(delta_pitch_rad * rotation_axis_unit)
        final_rotation = pitch_change_rotation * initial_rotation

    final_euler = final_rotation.as_euler('xyz', degrees=True)
    final_roll_deg = final_euler[0]
    final_pitch_deg_result = final_euler[1]
    final_yaw_deg = final_euler[2]

    place_position = place_ee_world + np.array([0, 0, 90])  # Increased z for clearance

    return pick_position, [roll_deg, initial_pitch_deg, yaw_deg], place_position, [final_roll_deg, final_pitch_deg_result, final_yaw_deg], approach_vector_unit

def pour():
#if __name__ == "__main__":

    realsense_streamer_81 = RealsenseStreamer(SERIAL_NO_81)
    realsense_streamer_56 = RealsenseStreamer(SERIAL_NO_56)

    while True:
        streamers = [realsense_streamer_81, realsense_streamer_56]
        results = [capture_and_process_rgbd(streamer) for streamer in streamers]
        pcds, realsense_imgs, depth_frames, depth_images = zip(*results)
        pcds, realsense_imgs, depth_frames, depth_images = np.array(pcds), np.array(realsense_imgs), np.array(depth_frames), np.array(depth_images)
        pixel_selector_81 = PixelSelector()
        pixel_selector_56 = PixelSelector()
        pixels_81, img_81 = pixel_selector_81.run(realsense_imgs[0])
        print(pixels_81)
        #cv2.imwrite(os.path.join(full_save_folder, f"cam81.png"), (img_81))

        pixels_56, img_56 = pixel_selector_56.run(realsense_imgs[1])
        print(pixels_56)
        #cv2.imwrite(os.path.join(full_save_folder, f"cam56.png"), (img_56))

        pixels = pixels_81
        pixels.extend(pixels_56)
        print(pixels)
        # pixels: 81_pick / average_pick, 56_pick, poured location
        
        '''cv2.namedWindow("Gaze Segmentation (Camera 81)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gaze Segmentation (Camera 81)", 640, 480)
        cv2.moveWindow("Gaze Segmentation (Camera 81)", 1800, 1050)
        cv2.namedWindow("Gaze Segmentation (Camera 56)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gaze Segmentation (Camera 56)", 640, 480)
        cv2.moveWindow("Gaze Segmentation (Camera 56)", 1800, 500)'''
        #print('pixel_81:', pixels[0])
        segmented_cam_imgs = [segment_image(realsense_imgs[0], np.array(pixels[0]))]
        mask_cam81 = segmented_cam_imgs[0]
        center_cam81 = None
        if np.any(mask_cam81):  # Check if the mask has any white pixels
            M = cv2.moments(mask_cam81.astype(np.uint8))
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center_cam81 = (cX, cY)
                print(f"Center of mask (Camera 81): {center_cam81}")

        cv2.imshow("Gaze Segmentation (Camera 81)", mask_cam81.astype(np.uint8) * 255)
        cv2.waitKey(0)
        #print('pixel_56:', pixels[1])
        segmented_cam_imgs.append(segment_image(realsense_imgs[1], np.array(pixels[1])))
        mask_cam56 = segmented_cam_imgs[1]
        center_cam56 = None
        if np.any(mask_cam56):  # Check if the mask has any white pixels
            M = cv2.moments(mask_cam56.astype(np.uint8))
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center_cam56 = (cX, cY)
                print(f"Center of mask (Camera 56): {center_cam56}")

        cv2.imshow("Gaze Segmentation (Camera 56)", mask_cam56.astype(np.uint8) * 255)
        cv2.waitKey(0)

        place_ee = streamers[1].deproject_pixel(pixels[2], depth_frames[1])
        place_ee_h = np.append(place_ee, 1).reshape(4, 1)
        place_ee_world = (TCR_56 @ place_ee_h).flatten() * 1000
        place_ee_world[2] += 200
        print("Place position (World Frame): ", place_ee_world)

        semantic_waypoint = streamers[0].deproject_pixel(center_cam81, depth_frames[0])
        waypoint_h = np.append(semantic_waypoint, 1).reshape(4, 1)
        transformed_waypoint = TCR_81 @ waypoint_h
        semantic_waypoint_world = transformed_waypoint.flatten() * 1000
        semantic_waypoint_world[2] += 100

        semantic_waypoint1 = streamers[1].deproject_pixel(center_cam56, depth_frames[1])
        waypoint_h1 = np.append(semantic_waypoint1, 1).reshape(4, 1)
        transformed_waypoint1 = TCR_56 @ waypoint_h1
        semantic_waypoint1_world = transformed_waypoint1.flatten() * 1000
        semantic_waypoint1_world[2] += 100

        averaged_pick_world = (semantic_waypoint_world + semantic_waypoint1_world) / 2
        print("Averaged pick position (World Frame): ", averaged_pick_world)

        pick_pos, initial_rpy, place_pos, final_rpy, approach_vector_unit = calculate_final_pose(
            place_ee_world, averaged_pick_world, initial_pitch_deg=-50, final_pitch_deg=-10
        )

        print("Initial RPY (Roll, Pitch, Yaw):", initial_rpy)
        print("Final RPY (Roll, Pitch, Yaw):", final_rpy)
        print("Pick Position:", pick_pos)
        print("Place Position:", place_pos)

        # Robot Control Sequence
        robot.grasp(None)
        #Approach the object 
        robot.move_to_ee_pose(pick_pos - 40.0 * Rotation.from_euler('xyz', (initial_rpy), degrees=True).as_matrix()[:, 2], initial_rpy)
        #Grab the object
        robot.move_to_ee_pose(pick_pos, initial_rpy)
        robot.grasp(10)
        time.sleep(1)
        #Move the object up vertically
        robot.move_to_ee_pose(pick_pos + np.array([0, 0, 50]), initial_rpy) # Move up slightly
        #Pour the object
        horizontal_displacement = -20.0  # Adjust this value as needed
        approach_vector_horizontal = approach_vector_unit[:2]
        approach_vector_horizontal_normalized = approach_vector_horizontal / np.linalg.norm(approach_vector_horizontal) if np.linalg.norm(approach_vector_horizontal) > 1e-6 else np.array([1, 0])
        horizontal_offset = np.array([approach_vector_horizontal_normalized[0] * horizontal_displacement,
                                      approach_vector_horizontal_normalized[1] * horizontal_displacement,
                                      0.0])
        pour_position = place_pos + horizontal_offset
        robot.move_to_ee_pose(pour_position, final_rpy)
        #Return to object previous position
        robot.move_to_ee_pose(pick_pos + np.array([0, 0, 50]), initial_rpy) # Move up slightly
        #Put down the object
        robot.move_to_ee_pose(pick_pos, initial_rpy)
        robot.grasp(None)
        time.sleep(1)
        #Go back to original position
        robot.move_to_ee_pose(pick_pos - 90.0 * Rotation.from_euler('xyz', (initial_rpy), degrees=True).as_matrix()[:, 2], initial_rpy)
        robot.go_home()
        
        sys.exit()
