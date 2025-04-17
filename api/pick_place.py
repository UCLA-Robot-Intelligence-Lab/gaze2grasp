import pyrealsense2 as rs
import cv2
import math
import os
import sys
import argparse
import time
import glob
import open3d as o3d
import numpy as np
import tensorflow.compat.v1 as tf
from scipy.spatial.transform import Rotation

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))
from contact_graspnet.contact_grasp_estimator import GraspEstimator
import contact_graspnet.config_utils
from multicam import XarmEnv
from rs_streamer import RealsenseStreamer
from visualizations.live_visualization import generate_and_visualize_grasps, segment_image, capture_and_process_rgbd

tf.disable_eager_execution()

# Constants
GRIPPER_SPEED, GRIPPER_FORCE, GRIPPER_MAX_WIDTH, GRIPPER_TOLERANCE = 0.1, 40, 0.08570, 0.01
SERIAL_NO_81 = '317422074281'  # Camera serial number
SERIAL_NO_56 = '317422075456'
# Load calibration data
transforms = np.load('calib/transforms.npy', allow_pickle=True).item()
TCR_81 = transforms[SERIAL_NO_81]['tcr']
TCR_56 = transforms[SERIAL_NO_56]['tcr']
TCR_81[:3, 3] /= 1000.0
TCR_56[:3, 3] /= 1000.0

base_link_color = [[1, 0.6, 0.8],  [1, 1, 0], [1, 0.5, 0], [0.4, 0, 0.8]]

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(f'GPUs: {physical_devices}')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))

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
    

base_link_color = [[1, 0.6, 0.8],  [1, 1, 0], [1, 0.5, 0], [0.4, 0, 0.8]]

def pick_place():
#if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='checkpoints/scene_test_2048_bs3_hor_sigma_001', help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]')#default='checkpoints/scene_2048_bs3_rad2_32', help='Log dir [default: checkpoints/scene_2048_bs3_rad2_32]') 
    #parser.add_argument('--np_path', default='test_data/7.npy', help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
    #parser.add_argument('--png_path', default='', help='Input data: depth map png in meters')
    parser.add_argument('--K', default=None, help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
    parser.add_argument('--z_range', default=[0.2,1.8], help='Z value threshold to crop the input point cloud')
    parser.add_argument('--local_regions', action='store_true', default=False, help='Crop 3D local regions around given segments.')
    parser.add_argument('--filter_grasps', action='store_true', default=False,  help='Filter grasp contacts according to segmap.')
    parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
    parser.add_argument('--forward_passes', type=int, default=1,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    parser.add_argument('--segmap_id', type=int, default=0,  help='Only return grasps of the given object id')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    FLAGS = parser.parse_args()

    global_config = contact_graspnet.config_utils.load_config(FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs)
    realsense_streamer_81 = RealsenseStreamer(SERIAL_NO_81)
    realsense_streamer_56 = RealsenseStreamer(SERIAL_NO_56)

    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()
    saver = tf.train.Saver(save_relative_paths=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    #print(config)
    #print(sess)
    #print('Session created: ', sess.list_devices())
    grasp_estimator.load_weights(sess, saver, FLAGS.ckpt_dir, mode='test')
    

    while True:
        base_folder = "vlm_images"
        while True:
            save_folder = input(f"Enter the folder name to save results (will be saved inside '{base_folder}'): ").strip()
            if not save_folder:
                save_folder = "default_results"  
            full_save_folder = os.path.join(base_folder, save_folder)
            if os.path.isdir(os.path.dirname(full_save_folder)):
                break
            print(f"Invalid folder name. Please try again.")
        os.makedirs(full_save_folder, exist_ok=True)
        print(f"Data will be saved in {full_save_folder}")

        os.makedirs('results', exist_ok=True)
        streamers = [realsense_streamer_81, realsense_streamer_56]
        results = [capture_and_process_rgbd(streamer) for streamer in streamers]
        pcds, realsense_imgs, depth_frames, depth_images = zip(*results)
        pcds, realsense_imgs, depth_frames, depth_images = np.array(pcds), np.array(realsense_imgs), np.array(depth_frames), np.array(depth_images)
        pixel_selector_81 = PixelSelector()
        pixel_selector_56 = PixelSelector()
        pixels_81, img_81 = pixel_selector_81.run(realsense_imgs[0])
        print(pixels_81)
        cv2.imwrite(os.path.join(full_save_folder, f"cam81.png"), (img_81))

        pixels_56, img_56 = pixel_selector_56.run(realsense_imgs[1])
        print(pixels_56)
        cv2.imwrite(os.path.join(full_save_folder, f"cam56.png"), (img_56))

        pixels = pixels_81
        pixels.extend(pixels_56)
        print(pixels)
        # pixels: 81_pick / average_pick, 56_pick, place
        
        cv2.namedWindow("Gaze Segmentation (Camera 81)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gaze Segmentation (Camera 81)", 640, 480)
        cv2.moveWindow("Gaze Segmentation (Camera 81)", 1800, 1050)
        cv2.namedWindow("Gaze Segmentation (Camera 56)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gaze Segmentation (Camera 56)", 640, 480)
        cv2.moveWindow("Gaze Segmentation (Camera 56)", 1800, 500)
        #print('pixel_81:', pixels[0])
        segmented_cam_imgs = [segment_image(realsense_imgs[0], np.array(pixels[0]))]
        
        cv2.imshow("Gaze Segmentation (Camera 81)", segmented_cam_imgs[0].astype(np.uint8) * 255)
        cv2.waitKey(0)
        #print('pixel_56:', pixels[1])
        segmented_cam_imgs.append(segment_image(realsense_imgs[1], np.array(pixels[1])))
        cv2.imshow("Gaze Segmentation (Camera 56)", segmented_cam_imgs[1].astype(np.uint8) * 255)
        cv2.waitKey(0)

        place_ee = streamers[1].deproject_pixel(pixels[2], depth_frames[1])
        place_ee_h = np.append(place_ee, 1).reshape(4, 1)
        place_ee = (TCR_56 @ place_ee_h).flatten() * 1000
        place_ee[2] = place_ee[2] + 220
        print("Place position (EE Frame): ", place_ee)
        # place_ee: averaged place position
        # 
        semantic_waypoint = streamers[0].deproject_pixel(pixels[0], depth_frames[0])
        waypoint_h = np.append(semantic_waypoint, 1).reshape(4, 1)
        transformed_waypoint = TCR_81 @ waypoint_h
        semantic_waypoint = transformed_waypoint.flatten() 
        # semantic_waypoint: averaged pick position
        
        intermediate_pos, positions, orientations, openings =  generate_and_visualize_grasps(semantic_waypoint, depth_images, segmented_cam_imgs, streamers, grasp_estimator, sess, full_save_folder)
        
        # To have the VLM select
        try:
            # consider moving this back up to the top with the rest of the imports??
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            sys.path.insert(0, parent_dir)

            from vlm_reason import vlm_select_pose
            optimal_pose = vlm_select_pose.gemini_select_pose(some_image) #idk where the image comes from
        except:
            pass

        position_fingertip = intermediate_pos[-1]
        position_ee = positions[-1]
        roll, pitch, yaw = orientations[-1]
        # Only move to the last pose
        robot.grasp(None)
        robot.go_home()
        robot.move_to_ee_pose(position_fingertip, [roll, pitch, yaw])
        robot.move_to_ee_pose(position_ee, [roll, pitch, yaw])
        robot.grasp(openings[-1])
        time.sleep(1)
        #robot.go_home()
        robot.move_to_ee_pose(np.array([475, 0, 245]), [roll, pitch, yaw])
        #robot.move_to_ee_pose(place_ee, np.array([-180, 0, 0]))
        robot.move_to_ee_pose(place_ee, [roll, pitch, yaw])
        robot.grasp(None)
        time.sleep(1)
        robot.go_home()

        sys.exit()
