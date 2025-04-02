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
from grasp_selector import find_closest_grasp, find_distinct_grasps
from visualize_gripper import visualize_gripper
import tensorflow.compat.v1 as tf
from segment.FastSAM.fastsam import FastSAM
from segment.SAMInference import select_from_sam_everything
from scipy.spatial.transform import Rotation
from contact_grasp_estimator import GraspEstimator
import config_utils
from multicam import XarmEnv

tf.disable_eager_execution()

# Constants
GRIPPER_SPEED, GRIPPER_FORCE, GRIPPER_MAX_WIDTH, GRIPPER_TOLERANCE = 0.1, 40, 0.08570, 0.01
SERIAL_NO = '317422074281'  # Camera serial number
# Load calibration data
transforms = np.load('calib/transforms.npy', allow_pickle=True).item()
TCR = transforms[SERIAL_NO]['tcr']
TCR[:3, 3] /= 1000.0

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(f'GPUs: {physical_devices}')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))

# Initialize robot
robot = XarmEnv()

class RealsenseStreamer():
    def __init__(self, serial_no=None):

        # in-hand : 317222072157
        # external: 317422075456

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if serial_no is not None:
            self.config.enable_device(serial_no)

        self.width = 640
        self.height = 480

        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)

        self.align_to_color = rs.align(rs.stream.color)

        # Start streaming
        self.pipe_profile = self.pipeline.start(self.config)

        profile = self.pipeline.get_active_profile()

        ## Configure depth sensor settings
        depth_sensor = profile.get_device().query_sensors()[0]
        depth_sensor.set_option(rs.option.enable_auto_exposure, 1)

        # DEPTH IMAGE IN MILLIMETERS NOT METERS  
        depth_sensor.set_option(rs.option.depth_units, 0.001) 
        
        preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
        for i in range(int(preset_range.max)):
            visualpreset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
            if visualpreset == "Default":
                depth_sensor.set_option(rs.option.visual_preset, i)

        color_sensor = profile.get_device().query_sensors()[1]
        color_sensor.set_option(rs.option.enable_auto_exposure, 1)

        self.serial_no = serial_no

        if self.serial_no == '317422075456':
            color_sensor.set_option(rs.option.exposure, 140)

        # Intrinsics & Extrinsics
        frames = self.pipeline.wait_for_frames()
        frames = self.align_to_color.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        self.depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        self.color_intrin = color_frame.profile.as_video_stream_profile().intrinsics

        self.depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
        self.colorizer = rs.colorizer()

        self.K = np.array([[self.depth_intrin.fx, 0, self.depth_intrin.ppx],
                           [0, self.depth_intrin.fy, self.depth_intrin.ppy],
                           [0, 0, 1]])

        #self.dec_filter = rs.decimation_filter()
        #self.spat_filter = rs.spatial_filter()
        #self.temp_filter = rs.temporal_filter()
        #self.hole_filling_filter = rs.hole_filling_filter()
        print("camera started")

    def deproject_pixel(self, px, depth_frame):
        u,v = px
        depth = depth_frame.get_distance(u,v)
        xyz = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [u,v], depth)
        return xyz

    def capture_rgb(self):
        color_frame = None
        while True:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame is not None:
                color_image = np.asanyarray(color_frame.get_data())
                break
        return color_image

    def filter_depth(self, depth_frame):
        filtered = depth_frame
        return filtered.as_depth_frame()

    def capture_rgbd(self):
        frame_error = True
        while frame_error:
            try:
                frames = self.align_to_color.process(frames)  
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                frame_error = False
            except:
                frames = self.pipeline.wait_for_frames()
                continue
        color_image = np.asanyarray(color_frame.get_data())
        depth_frame = self.filter_depth(depth_frame)
        depth_image = np.asanyarray(depth_frame.get_data())
        h, w = depth_image.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        u = u.flatten()
        v = v.flatten()

        # Compute 3D points for valid pixels
        points_3d = []
        for u_pixel, v_pixel in zip(u, v):
            point_3d = self.deproject_pixel((u_pixel, v_pixel), depth_frame)
            points_3d.append(np.array(point_3d))
        points_3d = np.array(points_3d)
        #print(points_3d.shape)
        #points_3d = np.vstack((points_3d, np.ones([1,points_3d.shape[1]])))
        #points_3d = ((np.eye(4).dot(points_3d)).T)[:, 0:3]
        colors = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB).reshape(points_3d.shape)/255.
        
        return points_3d, colors, color_frame, color_image, depth_frame, depth_image
    
    def stop_stream(self):
        self.pipeline.stop()

    def show_image(self, image):
        cv2.imshow('img', image)
        cv2.waitKey(0)

# Gripper control points
gripper_control_points = np.array([
    [0, 0, -0.2],  # Base point
    [0, 0, 0],      # Mid point
    [0, 0.05, 0.1], # Finger 1 tip
    [0, -0.05, 0.1],# Finger 2 tip
    [0, 0.05, 0],   # Finger 1 mid
    [0, -0.05, 0]   # Finger 2 mid
])

# Define the grasp line plot
mid_point = gripper_control_points[1]
grasp_line_plot = np.array([
    gripper_control_points[0],  # Base to mid
    mid_point,                  # Mid point
    gripper_control_points[2],  # Finger 1 tip
    gripper_control_points[3],  # Finger 2 tip
    gripper_control_points[4],  # Finger 1 mid
    gripper_control_points[5]   # Finger 2 mid
])

# Define connections for the gripper
connections = [np.array([
    [0, 1],  # Base to mid
    [1, 4],  # Mid to finger 1 mid
    [4, 2],  # Finger 1 mid to finger 1 tip
    [1, 5],  # Mid to finger 2 mid
    [5, 3]   # Finger 2 mid to finger 2 tip
])]

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
def visualize_gripper_with_cylinders(vis, grasps, pcd, connections, base_color):
    if grasps.size == 16:  
        grasps = [grasps]
    if type(base_color[0]) != list:
        base_color = [base_color]
    vis.clear_geometries()
    vis.add_geometry(pcd)
    for i, grasp in enumerate(grasps):
        #print(grasp)
        #print(base_color[i])
        gripper_control_points_closed = grasp_line_plot.copy()
        gripper_control_points_closed[2:6:2, 1] = np.sign(grasp_line_plot[2:6:2, 1]) * GRIPPER_MAX_WIDTH / 2
        
        R = np.array(grasp[:3, :3])
        t = np.array(grasp[:3, 3])
        pts = np.matmul(gripper_control_points_closed, R.T) + t
        for connection in connections[0]:
            start_point = pts[connection[0]]
            end_point = pts[connection[1]]
            cylinder = create_cylinder(start_point, end_point, 0.005, base_color[i])
            vis.add_geometry(cylinder)

def visualize_gripper_with_arm(vis, grasps, pcd, base_color):
    if grasps.size == 16:  
        grasps = [grasps]
    if type(base_color[0]) != list:
        base_color = [base_color]
    vis.clear_geometries()
    vis.add_geometry(pcd)
    for i, grasp in enumerate(grasps):
        gripper = visualize_gripper(grasp, base_color[i])
        for part in gripper:
            vis.add_geometry(part)
    
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
    points_3d, colors, _, realsense_img, depth_frame, depth_image = realsense_streamer.capture_rgbd()
    pcd = o3d.geometry.PointCloud()
    #print(points_3d.shape)
    #print(colors.shape)
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd, colors, realsense_img, depth_frame, depth_image

# Function to segment image using FastSAM
def segment_image(realsense_img, gaze):
    seg_model = FastSAM('./contact_graspnet/segment/FastSAM-s.pt')
    segmented_cam_img, _ = select_from_sam_everything(seg_model, [gaze], input_img=realsense_img, imgsz=640, iou=0.9, conf=0.4, max_distance=10, retina=True)
    return segmented_cam_img

# Function to predict grasps
def predict_grasps(grasp_estimator, sess, depth_image, segmented_cam_img, K, TCR, rgb):
    return grasp_estimator.predict_scene_grasps_from_depth_K_and_2d_seg(sess, depth_image, segmented_cam_img, K, TCR, rgb = rgb, local_regions=True, filter_grasps=False)

def set_camera_view_and_save_image(vis, extrinsic_matrix, output_filename):
    view_control = vis.get_view_control()
    param = view_control.convert_to_pinhole_camera_parameters()
    param.extrinsic = extrinsic_matrix
    view_control.convert_from_pinhole_camera_parameters(param)

    vis.update_renderer()
    vis.poll_events()
    time.sleep(1)

    # Capture and save the image
    float_buffer = vis.capture_screen_float_buffer()
    float_array = np.asarray(float_buffer)
    image_array = (255.0 * float_array).astype(np.uint8)
    cv2.imwrite(output_filename, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))

def process_grasp(grasp, save_folder, i, base_link_color):
    # Visualize and save gripper with cylinders
    if type(i) != int:
        color = base_link_color
    else:
        color = base_link_color[i]
    #print('color', color)
    visualize_gripper_with_cylinders(vis, grasp, pcd, connections, color)
    set_camera_view_and_save_image(vis, extrinsic_matrix, os.path.join(save_folder, f"grasp_lines{i}a.png"))
    set_camera_view_and_save_image(vis, extrinsic_matrix1, os.path.join(save_folder, f"grasp_lines{i}b.png"))

    # Visualize and save gripper with arm
    #visualize_gripper_with_arm(vis, grasp, pcd, color)
    #set_camera_view_and_save_image(vis, extrinsic_matrix, os.path.join(save_folder,f"grasp_arm{i}a.png"))
    #set_camera_view_and_save_image(vis, extrinsic_matrix1, os.path.join(save_folder,f"grasp_arm{i}b.png"))

    # Visualize and save gripper with axes
    visualize_gripper_with_axes(vis, grasp, pcd, color)
    set_camera_view_and_save_image(vis, extrinsic_matrix, os.path.join(save_folder,f"grasp_axes{i}a.png"))
    set_camera_view_and_save_image(vis, extrinsic_matrix1, os.path.join(save_folder,f"grasp_axes{i}b.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='checkpoints/scene_test_2048_bs3_hor_sigma_001', help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]')#default='checkpoints/scene_2048_bs3_rad2_32', help='Log dir [default: checkpoints/scene_2048_bs3_rad2_32]') 
    parser.add_argument('--np_path', default='test_data/7.npy', help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
    parser.add_argument('--png_path', default='', help='Input data: depth map png in meters')
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
    realsense_streamer = RealsenseStreamer(SERIAL_NO)

    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()
    saver = tf.train.Saver(save_relative_paths=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
   # print(config)
   # print(sess)
    #print('Session created: ', sess.list_devices())
    grasp_estimator.load_weights(sess, saver, FLAGS.ckpt_dir, mode='test')

    while True:
        _, colors, realsense_img, depth_frame, depth_image = capture_and_process_rgbd(realsense_streamer)
        gaze = np.array([348, 186])

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
        cv2.namedWindow("Gaze Segmentation (Camera)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gaze Segmentation (Camera)", 640, 480)
        cv2.moveWindow("Gaze Segmentation (Camera)", 1800, 1050)

        segmented_cam_img = segment_image(realsense_img, gaze)
        cv2.imshow("Gaze Segmentation (Camera)", segmented_cam_img.astype(np.uint8) * 255)
        cv2.waitKey(0)

        depth_image = depth_image / 1000
        pcd_new, pred_grasps_cam, _, _, pred_gripper_openings = predict_grasps(grasp_estimator, sess, depth_image, segmented_cam_img, realsense_streamer.K, TCR, rgb = realsense_img)
        
        #print('Predicted grasps:', pred_grasps_cam.items())
        print('No. Predicted grasps:', pred_grasps_cam[True].shape[0])
        extrinsic_matrix = np.load(f"./calib/extrinsic_{SERIAL_NO}.npy")
        extrinsic_matrix1 = np.load(f"./calib/extrinsic_{SERIAL_NO}_1.npy")
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        # Visualize all the predicted grasps
        visualize_gripper_with_cylinders(vis, pred_grasps_cam[True], pcd_new, connections, [[0, 0, 0] for _ in range(pred_grasps_cam[True].shape[0])])
        semantic_waypoint = realsense_streamer.deproject_pixel(gaze, depth_frame) 
        waypoint_h = np.append(semantic_waypoint, 1).reshape(4, 1)
        transformed_waypoint = TCR @ waypoint_h
        semantic_waypoint = transformed_waypoint.flatten() 
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(semantic_waypoint)
        sphere.paint_uniform_color([0,0,1])
        vis.add_geometry(sphere)
        set_camera_view_and_save_image(vis, extrinsic_matrix, os.path.join(full_save_folder, F"pred_grasp_lines_w_gaze.png"))  
        vis.run()
        # Select the options for VLM
        if pred_grasps_cam[True].shape[0] > 4:
            distinct_grasps, distinct_openings = find_distinct_grasps(pred_grasps_cam, pred_gripper_openings, semantic_waypoint, n_grasps=3, max_distance=0.2)
            closest_grasps, closest_opening = find_closest_grasp(pred_grasps_cam, pred_gripper_openings, semantic_waypoint)
            grasps = distinct_grasps
            openings = distinct_openings
            grasps.append(closest_grasps)
            openings.append(closest_opening)
            grasps = np.array(grasps)
            openings = np.array(openings)*1000
            print('openings:', openings)
        elif pred_grasps_cam[True].shape[0] == 4:
            grasps = pred_grasps_cam[True]
            openings = pred_gripper_openings[True]
            grasps = np.array(grasps)
            openings = np.array(openings)*1000
            print('openings:', openings)
        else:
            print("Insufficient grasp predictions...Exiting")
            break
        pcd = pcd_new
        # Visualizing single grasps
        base_link_color =[[1, 0.6, 0.8], [0.4, 0, 0.8], [1, 0.5, 0], [1, 1, 0]]
        positions, orientations = [], []
        for i, grasp in enumerate(grasps):
            #print('Loop: ', i)
            process_grasp(grasp, full_save_folder, i, base_link_color)
            
            # Move robot to the grasp pose
            print("Moving to pose")
            robot.grasp(None)
            robot.go_home()
            #position_fingertip = TCR[:3, :3] @ (1000.0 * grasp[:3, 3]) + TCR[:3, 3]
            #rotation_matrix_rob = TCR[:3, :3] @ grasp[:3, :3]
            position_fingertip = 1000.0 *grasp[:3, 3]
            rotation_matrix_rob = grasp[:3, :3]
            approach_dir_base = rotation_matrix_rob[:, 2]
            position_ee = position_fingertip #+ 90.0 * approach_dir_base
            rot = Rotation.from_matrix(rotation_matrix_rob)
            [yaw, pitch, roll] = rot.as_euler('ZYX', degrees=True)
            print("Adjusted Position (EE Frame): ", position_ee)
            print("Orientation (EE Frame): ", [roll, pitch, yaw])
            positions.append(position_ee)
            orientations.append([roll, pitch, yaw])
            robot.move_to_ee_pose(position_ee, [roll, pitch, yaw])
            robot.grasp(openings[i])


        # Visualizing all grasps
        process_grasp(grasps, full_save_folder, '_all_', base_link_color)

        # Saving all the data
        data = np.array(list(zip(grasps, positions, orientations)), dtype=object)
        np.save(os.path.join(full_save_folder, "grasp_data.npy"), data)
        
        vis.destroy_window()

        sys.exit()
