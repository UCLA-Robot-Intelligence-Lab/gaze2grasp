import pyrealsense2 as rs
import cv2
import math
import os
import sys
import argparse
import time
import open3d as o3d
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from load_np_file import load_np, full_path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))
from contact_graspnet.grasp_selector import find_closest_grasp, find_distinct_grasps
#from meshes.visualize_gripper import visualize_gripper
import tensorflow.compat.v1 as tf
from scipy.spatial.transform import Rotation
from contact_graspnet.contact_grasp_estimator import GraspEstimator
import contact_graspnet.config_utils as config_utils
from multicam import XarmEnv
from rs_streamer import RealsenseStreamer
from visualizations.live_visualization import segment_image, capture_and_process_rgbd, predict_grasps, ensure_camera_up, process_grasp


tf.disable_eager_execution()

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

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(f'GPUs: {physical_devices}')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

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

    
def pick_micro_contact_graspnet():
# if __name__ == "__main__":
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

    global_config = config_utils.load_config(FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs)
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
        segmented_cam_imgs.append(segment_image(realsense_imgs[1], np.array(pixels[2])))
        cv2.imshow("Gaze Segmentation (Camera 56)", segmented_cam_imgs[1].astype(np.uint8) * 255)
        cv2.waitKey(0)

        semantic_waypoint = streamers[0].deproject_pixel(pixels[0], depth_frames[0])
        waypoint_h = np.append(semantic_waypoint, 1).reshape(4, 1)
        transformed_waypoint = TCR_81 @ waypoint_h
        semantic_waypoint_world = transformed_waypoint.flatten() 
        print("Pick position (World Frame): ", semantic_waypoint_world)
        semantic_waypoint1 = streamers[1].deproject_pixel(pixels[2], depth_frames[1])
        waypoint_h1 = np.append(semantic_waypoint1, 1).reshape(4, 1)
        transformed_waypoint1 = TCR_56 @ waypoint_h1
        semantic_waypoint1_world = transformed_waypoint1.flatten()
        print("Pick1 position (World Frame): ", semantic_waypoint1_world)

        averaged_pick_world = (semantic_waypoint_world + semantic_waypoint1_world) / 2
        print("Averaged position (World Frame): ", averaged_pick_world)
        # semantic_waypoint: averaged pick position

        place_ee = streamers[0].deproject_pixel(pixels[1], depth_frames[0])
        place_ee_h = np.append(place_ee, 1).reshape(4, 1)
        place_ee_world = (TCR_81 @ place_ee_h).flatten() * 1000
        place_ee_world[2] += 200
        print("Place position (World Frame): ", place_ee_world)
        
        depth_images = np.array(depth_images) / 1000  # Convert list to array first
        merged_pcd, transformed_pcds, pred_grasps_cam, _, _, pred_gripper_openings = predict_grasps(grasp_estimator, sess, depth_images, np.array(segmented_cam_imgs), np.array([streamers[0].K, streamers[1].K]), np.array([TCR_81, TCR_56]), rgb = realsense_imgs)
        print('Predicted grasps:', pred_grasps_cam[True].shape[0])
        
        vis = o3d.visualization.Visualizer()
        window_width = 4988
        window_height = 2742
        vis.create_window(width=window_width, height=window_height)
    
        '''visualize_gripper_with_cylinders(vis, pred_grasps_cam[True], merged_pcd, connections, [[0, 0, 0] for _ in range(pred_grasps_cam[True].shape[0])])
        
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(averaged_pick_world)
        sphere.paint_uniform_color([0,0,1])
        vis.add_geometry(sphere)
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        vis.add_geometry(origin)
        vis.run()
        set_camera_view_and_save_image(vis, intrinsic, extrinsic_matrix, os.path.join(full_save_folder, F"pred_grasp_lines_w_gaze.png"), center = averaged_pick_world.copy())  '''
        intermediate_pos = []
        # Select the options for VLM
        distinct_grasps, distinct_openings = find_distinct_grasps(pred_grasps_cam, pred_gripper_openings, averaged_pick_world, n_grasps=3, max_distance=0.2)
        closest_grasps, closest_opening = find_closest_grasp(pred_grasps_cam, pred_gripper_openings, averaged_pick_world)
        grasps = distinct_grasps
        openings = distinct_openings
        grasps.append(closest_grasps)
        openings.append(closest_opening)
        grasps = np.array(grasps)
        openings = np.array(openings)*1000
        print(openings)
        

        # Visualizing single grasps
        positions, orientations = [], []
        for i, grasp in enumerate(grasps):
            # Visualizing with merged point cloud
            os.makedirs(os.path.join(full_save_folder, "pcd_combined"), exist_ok=True)
            process_grasp(vis,merged_pcd, grasp, os.path.join(full_save_folder, "pcd_combined"), i, base_link_color, center = averaged_pick_world)
            # Visualizing with individual point cloud
            os.makedirs(os.path.join(full_save_folder, "pcd81"), exist_ok=True)
            os.makedirs(os.path.join(full_save_folder, "pcd56"), exist_ok=True)
            process_grasp(vis,transformed_pcds[0], grasp, os.path.join(full_save_folder, "pcd81"), i, base_link_color, view = "81", center = averaged_pick_world)
            process_grasp(vis,transformed_pcds[1], grasp, os.path.join(full_save_folder, "pcd56"), i, base_link_color, view = "56", center = averaged_pick_world)
            
            # Move robot to the grasp pose
            position_fingertip = 1000 * grasp[:3, 3]
            position_fingertip[2] = position_fingertip[2]+157
            rotation_matrix_rob = grasp[:3, :3]
            rotation_matrix_rob = ensure_camera_up(rotation_matrix_rob)
            approach_dir_base = rotation_matrix_rob[:, 2]
            position_ee = position_fingertip + 90.0 * approach_dir_base
            rot = Rotation.from_matrix(rotation_matrix_rob)
            [yaw, pitch, roll] = rot.as_euler('ZYX', degrees=True)
            print("Approach Position (EE Frame): ", position_fingertip)
            print("Adjusted Position (EE Frame): ", position_ee)
            print("Orientation (EE Frame): ", [roll, pitch, yaw])
            intermediate_pos.append(position_fingertip)
            positions.append(position_ee)
            orientations.append([roll, pitch, yaw])

        k = int(input(f"Enter the index of grasp: ").strip())
        position_ee = positions[-k]
        roll, pitch, yaw = orientations[-k]
        position_fingertip = intermediate_pos[-k]
        
        initial_rpy = np.array([roll, pitch, yaw])
        above_door_pos = position_ee.copy()
        above_door_pos[2] = 430
        place_rpy = np.array([139,-12,50])
        place_pos = np.array([507, -227, 143])

        #initial_rpy angle can change because object changes pose after grasp

        # Robot Control Sequence
        robot.grasp(None)
        #Approach the object
        robot.move_to_ee_pose(position_fingertip, initial_rpy)
        #Grab the object
        robot.move_to_ee_pose(position_ee, initial_rpy)
        robot.grasp(3)
        time.sleep(2)
        #Move the object above the door
        robot.move_to_ee_pose(above_door_pos, initial_rpy)        
        robot.move_to_ee_pose(np.array([435, 0, 450]), place_rpy)
        robot.move_to_ee_pose(np.array([260, -150, 250]), place_rpy)
        #Move object into microwave
        #robot.move_to_ee_pose(place_ee_world, initial_rpy)
        robot.move_to_ee_pose(place_pos,place_rpy)
        robot.grasp(None)
        time.sleep(2)
        #Go to home
        robot.move_to_ee_pose(place_pos - 90 * Rotation.from_euler('xyz', (place_rpy), degrees=True).as_matrix()[:, 2], place_rpy)
        #robot.move_to_ee_pose(place_ee_world - 90 * Rotation.from_euler('xyz', (place_rpy), degrees=True).as_matrix()[:, 2], place_rpy)
        #robot.move_to_ee_pose(place_ee_world - 90 * Rotation.from_euler('xyz', (initial_rpy), degrees=True).as_matrix()[:, 2], np.array([180, 0, 0]))
        robot.go_home()


        # Visualizing all grasps
        process_grasp(vis,merged_pcd, grasps, os.path.join(full_save_folder, "pcd_combined"), '_all_', base_link_color, center = averaged_pick_world)
        process_grasp(vis,transformed_pcds[0], grasps, os.path.join(full_save_folder, "pcd81"), '_all_', base_link_color, view = "81", center = averaged_pick_world)
        process_grasp(vis,transformed_pcds[1], grasps, os.path.join(full_save_folder, "pcd56"), '_all_', base_link_color, view = "56", center = averaged_pick_world)

        # Saving all the data
        data = np.array(list(zip(grasps, positions, orientations)), dtype=object)
        np.save(os.path.join(full_save_folder, "grasp_data.npy"), data)

        
        vis.destroy_window()

        sys.exit()
