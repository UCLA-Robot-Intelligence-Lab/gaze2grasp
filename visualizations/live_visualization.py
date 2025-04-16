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
from contact_graspnet.grasp_selector import find_closest_grasp, find_distinct_grasps
#from meshes.visualize_gripper import visualize_gripper
import tensorflow.compat.v1 as tf
from segment.FastSAM.fastsam import FastSAM
from segment.SAMInference import select_from_sam_everything
from scipy.spatial.transform import Rotation
from contact_graspnet.contact_grasp_estimator import GraspEstimator
import contact_graspnet.config_utils
from multicam import XarmEnv
from rs_streamer import RealsenseStreamer

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
    
gripper_control_points = np.array([
    [0, 0, -0.157],  # Base point
    [0, 0, -0.005],   # Mid point
    [0, 0.025, 0.035], # Finger 1 tip
    [0, -0.025, 0.035], # Finger 2 tip
    [0, 0.025, 0],     # Finger 1 mid
    [0, -0.025, 0],    # Finger 2 mid
    [0, 0, 0.05]       # Extension endpoint 
])

gripper_control_points[:, 2] += 0.06

grasp_line_plot = np.array([
    gripper_control_points[0],  # Base to mid
    gripper_control_points[1],  # Mid point
    gripper_control_points[2],  # Finger 1 tip
    gripper_control_points[3],  # Finger 2 tip
    gripper_control_points[4],  # Finger 1 mid
    gripper_control_points[5],  # Finger 2 mid
    gripper_control_points[6]   # Extension endpoint
])

connections = [np.array([
    [0, 1],  # Base to mid
    [1, 4],  # Mid to finger 1 mid
    [4, 2],  # Finger 1 mid to finger 1 tip
    [1, 5],  # Mid to finger 2 mid
    [5, 3],  # Finger 2 mid to finger 2 tip
    [1, 6]   # Mid point to extension endpoint
])]

def ensure_camera_up(R):
    """Ensures the gripper's camera is on the upper side by flipping Z-rotation if needed."""
    gripper_up_dir = R[:, 1]  # Y-axis of the gripper
    world_up_dir = np.array([0, 0, 1])  # Assuming Z-up world
    
    if np.dot(gripper_up_dir, world_up_dir) < 0:
        R_flip_z = np.array([
            [-1,  0,  0],
            [ 0, -1,  0],
            [ 0,  0,  1]
        ])
        print("Flipping Z-axis of the gripper for gripper camera to face up")
        return R @ R_flip_z  # Apply local Z-flip
    else:
        return R

# Function to create a cylinder for visualization
def create_cylinder(start_point, end_point, radius, color):
    cylinder_vector = end_point - start_point
    cylinder_length = np.linalg.norm(cylinder_vector)
    cylinder_midpoint = (start_point + end_point) / 2
    cylinder_direction = cylinder_vector / cylinder_length

    cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(radius, cylinder_length, resolution=4, split=1)
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, cylinder_direction)
    cylinder_direction = cylinder_direction.flatten()
    rotation_angle = np.arccos(np.dot(z_axis, cylinder_direction))

    if np.linalg.norm(rotation_axis) > 0:
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
            rotation_angle * rotation_axis / np.linalg.norm(rotation_axis)
        )
        cylinder_segment.rotate(rotation_matrix, center=(0, 0, 0))
    cylinder_segment.translate(cylinder_midpoint)
    color = np.array(color, dtype=np.float64).reshape(3, 1)
    cylinder_segment.paint_uniform_color(color)
    return cylinder_segment

# Function to visualize gripper
def visualize_gripper_with_cylinders(vis, grasps, pcd, connections=connections, base_color=base_link_color):
    if grasps.size == 16:  
        grasps = [grasps]
    if type(base_color[0]) != list:
        base_color = [base_color]
    vis.clear_geometries()
    vis.add_geometry(pcd)
    
    for i, grasp in enumerate(grasps):
        # Create closed gripper points
        gripper_control_points_closed = grasp_line_plot.copy()
        gripper_control_points_closed[2:6:2, 1] = np.sign(grasp_line_plot[2:6:2, 1]) * GRIPPER_MAX_WIDTH / 2
        
        # Apply grasp transformation
        R = np.array(grasp[:3, :3])
        t = np.array(grasp[:3, 3])
        pts = np.matmul(gripper_control_points_closed, R.T) + t
        
        # Draw all connections
        for connection in connections[0]:
            start_point = pts[connection[0]]
            end_point = pts[connection[1]]
            
            # Use thinner cylinder for the extension line (connection [1,6])
            radius = 0.002 if connection[1] == 6 else 0.005
            
            cylinder = create_cylinder(start_point, end_point, radius, base_color[i])
            vis.add_geometry(cylinder)

'''def visualize_gripper_with_arm(vis, grasps, pcd, base_color):
    if grasps.size == 16:  
        grasps = [grasps]
    if type(base_color[0]) != list:
        base_color = [base_color]
    vis.clear_geometries()
    vis.add_geometry(pcd)
    for i, grasp in enumerate(grasps):
        gripper = visualize_gripper(grasp, base_color[i])
        for part in gripper:
            vis.add_geometry(part)'''
    
def visualize_gripper_with_axes(vis, grasps, pcd, base_color):
    if grasps.size == 16:  
        grasps = [grasps]
    if type(base_color[0]) != list:
        base_color = [base_color]
    vis.clear_geometries()
    vis.add_geometry(pcd)
    for i, grasp in enumerate(grasps):
        R = np.array(grasp[:3, :3])  # Extract rotation
        t = np.array(grasp[:3, 3])   # Extract translation (position)
        approach_dir_base = R[:, 2]
        t = t + 0.06 * approach_dir_base

        # Create coordinate frame and sphere for visualization
        grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=t)
        grasp_frame.rotate(R, center=t)  # Apply the grasp rotation
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(t)
        sphere.paint_uniform_color(base_color[i])
        vis.add_geometry(grasp_frame)
        vis.add_geometry(sphere)

# Function to capture and process RGBD data
def capture_and_process_rgbd(realsense_streamer):
    points_3d, colors, _, realsense_img, depth_frame, depth_image = realsense_streamer.capture_rgbdpc()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd, realsense_img, depth_frame, depth_image

# Function to segment image using FastSAM
def segment_image(realsense_img, gaze):
    seg_model = FastSAM('./contact_graspnet/segment/FastSAM-s.pt')
    segmented_cam_img, _ = select_from_sam_everything(seg_model, [gaze], input_img=realsense_img, imgsz=640, iou=0.9, conf=0.4, max_distance=5, max_mask_ratio = 0.2, retina=True)
    return segmented_cam_img

# Function to predict grasps
def predict_grasps(grasp_estimator, sess, depth_image, segmented_cam_img, K, TCR, rgb):
    return grasp_estimator.predict_scene_grasps_from_depth_K_and_2d_seg(sess, depth_image, segmented_cam_img, K, TCR, rgb = rgb, local_regions=True, filter_grasps=True)

def set_camera_view_and_save_image(vis, intrinsic, extrinsic_matrix, output_filename):
    view_control = vis.get_view_control()
    param = view_control.convert_to_pinhole_camera_parameters()
    param.intrinsic = intrinsic
    param.extrinsic = extrinsic_matrix
    view_control.convert_from_pinhole_camera_parameters(param)

    vis.update_renderer()
    vis.poll_events()
    #time.sleep(0.2)

    # Capture and save the image
    float_buffer = vis.capture_screen_float_buffer()
    float_array = np.asarray(float_buffer)
    image_array = (255.0 * float_array).astype(np.uint8)
    height, width, _ = image_array.shape

    crop_height = 480
    crop_width = 854

    center_y = height // 2
    center_x = width // 2

    ymin = center_y - crop_height // 2
    ymax = center_y + crop_height // 2
    xmin = center_x - crop_width // 2
    xmax = center_x + crop_width // 2

    ymin = max(0, ymin)
    ymax = min(height, ymax)
    xmin = max(0, xmin)
    xmax = min(width, xmax)

    cropped_image_array = image_array[ymin:ymax, xmin:xmax]

    cv2.imwrite(output_filename, cv2.cvtColor(cropped_image_array, cv2.COLOR_RGB2BGR))

def process_grasp(vis, merged_pcds, grasp, save_folder, i, base_link_color, view = True):
    # Visualize and save gripper with cylinders
    if type(i) != int:
        color = base_link_color
    else:
        color = base_link_color[i]
    if view == "81":
        visualize_gripper_with_cylinders(vis, grasp, merged_pcds, connections, color)
        set_camera_view_and_save_image(vis, intrinsic2, extrinsic_matrix2, os.path.join(save_folder, f"grasp_lines{i}.png"))
        visualize_gripper_with_axes(vis, grasp, merged_pcds, color)
        set_camera_view_and_save_image(vis, intrinsic2, extrinsic_matrix2, os.path.join(save_folder,f"grasp_axes{i}.png"))
    elif view == "56":
        visualize_gripper_with_cylinders(vis, grasp, merged_pcds, connections, color)
        set_camera_view_and_save_image(vis, intrinsic3, extrinsic_matrix3, os.path.join(save_folder, f"grasp_lines{i}.png"))
        visualize_gripper_with_axes(vis, grasp, merged_pcds, color)
        set_camera_view_and_save_image(vis, intrinsic3, extrinsic_matrix3, os.path.join(save_folder,f"grasp_axes{i}.png"))
    else:
        visualize_gripper_with_cylinders(vis, grasp, merged_pcds, connections, color)
        set_camera_view_and_save_image(vis, intrinsic, extrinsic_matrix, os.path.join(save_folder, f"grasp_lines{i}a.png"))
        set_camera_view_and_save_image(vis, intrinsic1, extrinsic_matrix1, os.path.join(save_folder, f"grasp_lines{i}b.png"))
        set_camera_view_and_save_image(vis, intrinsic2, extrinsic_matrix2, os.path.join(save_folder, f"grasp_lines{i}c.png"))
        set_camera_view_and_save_image(vis, intrinsic3, extrinsic_matrix3, os.path.join(save_folder, f"grasp_lines{i}d.png"))

        # Visualize and save gripper with arm
        #visualize_gripper_with_arm(vis, grasp, merged_pcds, color)
        #set_camera_view_and_save_image(vis, extrinsic_matrix, os.path.join(save_folder,f"grasp_arm{i}a.png"))
        #set_camera_view_and_save_image(vis, extrinsic_matrix1, os.path.join(save_folder,f"grasp_arm{i}b.png"))

        # Visualize and save gripper with axes
        visualize_gripper_with_axes(vis, grasp, merged_pcds, color)
        set_camera_view_and_save_image(vis, intrinsic, extrinsic_matrix, os.path.join(save_folder,f"grasp_axes{i}a.png"))
        set_camera_view_and_save_image(vis, intrinsic1, extrinsic_matrix1, os.path.join(save_folder,f"grasp_axes{i}b.png"))
        set_camera_view_and_save_image(vis, intrinsic2, extrinsic_matrix2, os.path.join(save_folder,f"grasp_axes{i}c.png"))
        set_camera_view_and_save_image(vis, intrinsic3, extrinsic_matrix3, os.path.join(save_folder,f"grasp_axes{i}d.png"))
    
def load_intrinsic_matrix(file_path):
    """Loads the intrinsic matrix from a .npy file."""
    loaded_data = np.load(file_path, allow_pickle=True).item()
    width = loaded_data["width"]
    height = loaded_data["height"]
    intrinsic_matrix = loaded_data["intrinsic_matrix"]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2])
    return intrinsic

extrinsic_matrix = np.load(f"./calib/extrinsic_combined1.npy")
extrinsic_matrix1 = np.load(f"./calib/extrinsic_combined2.npy")
extrinsic_matrix2 = np.load(f"./calib/extrinsic_combined3.npy")
extrinsic_matrix3 = np.load(f"./calib/extrinsic_combined4.npy")
intrinsic = load_intrinsic_matrix(f"./calib/intrinsic1.npy")
intrinsic1 = load_intrinsic_matrix(f"./calib/intrinsic2.npy")
intrinsic2 = load_intrinsic_matrix(f"./calib/intrinsic3.npy")
intrinsic3 = load_intrinsic_matrix(f"./calib/intrinsic4.npy")

base_link_color = [[1, 0.6, 0.8],  [1, 1, 0], [1, 0.5, 0], [0.4, 0, 0.8]]

def generate_and_visualize_grasps(semantic_waypoint, depth_images, segmented_cam_imgs, streamers, grasp_estimator, sess, full_save_folder):
    depth_images = np.array(depth_images) / 1000  # Convert list to array first
    merged_pcd, transformed_pcds, pred_grasps_cam, _, _, pred_gripper_openings = predict_grasps(grasp_estimator, sess, depth_images, np.array(segmented_cam_imgs), np.array([streamers[0].K, streamers[1].K]), np.array([TCR_81, TCR_56]), rgb = realsense_imgs)
    print('Predicted grasps:', pred_grasps_cam[True].shape[0])


    vis = o3d.visualization.Visualizer()
    window_width = 4988
    window_height = 2742
    vis.create_window(width=window_width, height=window_height)
    visualize_gripper_with_cylinders(vis, pred_grasps_cam[True], merged_pcd, connections, [[0, 0, 0] for _ in range(pred_grasps_cam[True].shape[0])])
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    sphere.translate(semantic_waypoint)
    sphere.paint_uniform_color([0,0,1])
    vis.add_geometry(sphere)
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis.add_geometry(origin)
    vis.run()
    set_camera_view_and_save_image(vis, intrinsic, extrinsic_matrix, os.path.join(full_save_folder, F"pred_grasp_lines_w_gaze.png")) 


    distinct_grasps, distinct_openings = find_distinct_grasps(pred_grasps_cam, pred_gripper_openings, semantic_waypoint, n_grasps=3, max_distance=0.2)
    closest_grasps, closest_opening = find_closest_grasp(pred_grasps_cam, pred_gripper_openings, semantic_waypoint)
    grasps = distinct_grasps
    openings = distinct_openings
    grasps.append(closest_grasps)
    openings.append(closest_opening)
    grasps = np.array(grasps)
    openings = np.array(openings)*1000
    intermediate_pos, positions, orientations = [], [], []
    '''vis = o3d.visualization.Visualizer()
    window_width = 4988
    window_height = 2742
    vis.create_window(width=window_width, height=window_height)'''
    for i, grasp in enumerate(grasps):
        # Visualizing with merged point cloud
        #os.makedirs(os.path.join(full_save_folder, "pcd_combined"), exist_ok=True)
        #process_grasp(vis, merged_pcd, grasp, os.path.join(full_save_folder, "pcd_combined"), i, base_link_color)
        # Visualizing with individual point cloud
        os.makedirs(os.path.join(full_save_folder, "pcd81"), exist_ok=True)
        os.makedirs(os.path.join(full_save_folder, "pcd56"), exist_ok=True)
        process_grasp(vis, transformed_pcds[0], grasp, os.path.join(full_save_folder, "pcd81"), i, base_link_color, view = "81")
        process_grasp(vis, transformed_pcds[1], grasp, os.path.join(full_save_folder, "pcd56"), i, base_link_color, view = "56")
        
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
    # Visualizing all grasps
    #process_grasp(vis, merged_pcd, grasps, os.path.join(full_save_folder, "pcd_combined"), '_all_', base_link_color)
    process_grasp(vis, transformed_pcds[0], grasps, os.path.join(full_save_folder, "pcd81"), '_all_', base_link_color, view = "81")
    process_grasp(vis, transformed_pcds[1], grasps, os.path.join(full_save_folder, "pcd56"), '_all_', base_link_color, view = "56")

    # Saving all the data
    data = np.array(list(zip(grasps, intermediate_pos, positions, orientations)), dtype=object)
    np.save(os.path.join(full_save_folder, "grasp_data.npy"), data)
    vis.destroy_window()

    return intermediate_pos, positions, orientations, openings
    

if __name__ == "__main__":
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
        
        '''depth_images = np.array(depth_images) / 1000  # Convert list to array first
        merged_pcd, transformed_pcds, pred_grasps_cam, _, _, pred_gripper_openings = predict_grasps(grasp_estimator, sess, depth_images, np.array(segmented_cam_imgs), np.array([streamers[0].K, streamers[1].K]), np.array([TCR_81, TCR_56]), rgb = realsense_imgs)
        print('Predicted grasps:', pred_grasps_cam[True].shape[0])
        
        #vis.create_window()
        # Visualize all the predicted grasps
        visualize_gripper_with_cylinders(vis, pred_grasps_cam[True], merged_pcd, connections, [[0, 0, 0] for _ in range(pred_grasps_cam[True].shape[0])])
        vis = o3d.visualization.Visualizer()
        window_width = 4988
        window_height = 2742
        vis.create_window(width=window_width, height=window_height)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(semantic_waypoint)
        sphere.paint_uniform_color([0,0,1])
        vis.add_geometry(sphere)
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        vis.add_geometry(origin)
        vis.run()
        set_camera_view_and_save_image(vis, intrinsic, extrinsic_matrix, os.path.join(full_save_folder, F"pred_grasp_lines_w_gaze.png"))  
        
        # Select the options for VLM
        distinct_grasps, distinct_openings = find_distinct_grasps(pred_grasps_cam, pred_gripper_openings, semantic_waypoint, n_grasps=3, max_distance=0.2)
        closest_grasps, closest_opening = find_closest_grasp(pred_grasps_cam, pred_gripper_openings, semantic_waypoint)
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
            process_grasp(vis,merged_pcd, grasp, os.path.join(full_save_folder, "pcd_combined"), i, base_link_color)
            # Visualizing with individual point cloud
            os.makedirs(os.path.join(full_save_folder, "pcd81"), exist_ok=True)
            os.makedirs(os.path.join(full_save_folder, "pcd56"), exist_ok=True)
            process_grasp(vis,transformed_pcds[0], grasp, os.path.join(full_save_folder, "pcd81"), i, base_link_color, view = "81")
            process_grasp(vis,transformed_pcds[1], grasp, os.path.join(full_save_folder, "pcd56"), i, base_link_color, view = "56")
            

            # Move robot to the grasp pose
            print("Moving to pose")
            #robot.grasp(None)
            #robot.go_home()
            #position_fingertip = TCR[:3, :3] @ (1000.0 * grasp[:3, 3]) + TCR[:3, 3]
            #rotation_matrix_rob = TCR[:3, :3] @ grasp[:3, :3]
            print("Move to Grasp matrix: ", grasp)
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
            positions.append(position_ee)
            orientations.append([roll, pitch, yaw])
            #robot.move_to_ee_pose(position_fingertip, [roll, pitch, yaw])
            #robot.move_to_ee_pose(position_ee, [roll, pitch, yaw])
            #robot.grasp(openings[i])
            #robot.grasp(None)
            #robot.move_to_ee_pose(position_fingertip, [roll, pitch, yaw])
        '''
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


        '''# Visualizing all grasps
        process_grasp(vis,merged_pcd, grasps, os.path.join(full_save_folder, "pcd_combined"), '_all_', base_link_color)
        process_grasp(vis,transformed_pcds[0], grasps, os.path.join(full_save_folder, "pcd81"), '_all_', base_link_color, view = "81")
        process_grasp(vis,transformed_pcds[1], grasps, os.path.join(full_save_folder, "pcd56"), '_all_', base_link_color, view = "56")

        # Saving all the data
        data = np.array(list(zip(grasps, positions, orientations)), dtype=object)
        np.save(os.path.join(full_save_folder, "grasp_data.npy"), data)

        
        vis.destroy_window()'''

        sys.exit()
