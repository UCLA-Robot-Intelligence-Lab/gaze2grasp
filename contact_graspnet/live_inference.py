import pyrealsense2 as rs
import cv2
import math
import os
import sys
import argparse
import time
import glob
import open3d as o3d
from grasp_selector import find_closest_grasp, find_distinct_grasps
from visualize_gripper import visualize_gripper
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import numpy as np

from segment.FastSAM.fastsam import FastSAM
from segment.SAMInference import select_from_sam_everything

GRIPPER_SPEED, GRIPPER_FORCE, GRIPPER_MAX_WIDTH, GRIPPER_TOLERANCE = 0.1, 40, 0.08570, 0.01
#serial_no = '317422075456' #(camera further from robot arm)
serial_no = '317422074281'

#transforms = np.load(f'calib/transforms_{serial_no}.npy', allow_pickle=True).item()
transforms = np.load(f'calib/transforms.npy', allow_pickle=True).item()
TCR = transforms[serial_no]['tcr']

from multicam import XarmEnv

# from rs_streamer import RealsenseStreamer
from scipy.spatial.transform import Rotation 
from calib_utils.linalg_utils import transform

robot = XarmEnv()

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

def transform_rotation_camera_to_robot_roll_yaw_pitch(rotation_cam, TCR):
    """
    Transforms a rotation from camera frame to robot base frame and returns Roll-Yaw-Pitch.
    """
    # Convert camera rotation to quaternion
    r_cam = Rotation.from_matrix(rotation_cam)
    q_cam = r_cam.as_quat()

    # Convert TCR rotation to quaternion
    r_tcr = Rotation.from_matrix(TCR[:3, :3])
    q_tcr = r_tcr.as_quat()

    # Multiply quaternions
    q_rob = Rotation.from_quat(q_tcr) * Rotation.from_quat(q_cam)
    q_rob = q_rob.as_quat()

    # Convert robot rotation to Roll-Yaw-Pitch (ZYX Euler angles)
    r_rob = Rotation.from_quat(q_rob)
    roll_yaw_pitch = r_rob.as_euler('zyx', degrees=True)

    roll, pitch, yaw = roll_yaw_pitch[2], roll_yaw_pitch[1], roll_yaw_pitch[0]
    return np.array([roll, pitch, yaw]) # Return as (roll, pitch, yaw)

def get_extrinsic_matrix(camera_position, lookat, front, up):
    """
    Constructs an extrinsic matrix from camera position, lookat, front, and up vectors.

    Args:
        camera_position: NumPy array (3,) representing the camera's position.
        lookat: NumPy array (3,) representing the lookat point.
        front: NumPy array (3,) representing the front vector.
        up: NumPy array (3,) representing the up vector.

    Returns:
        NumPy array (4, 4) representing the extrinsic matrix.
    """

    # Normalize vectors
    front = front / np.linalg.norm(front)
    up = up / np.linalg.norm(up)

    # Calculate right vector
    right = np.cross(front, up)
    right = right / np.linalg.norm(right)

    # Re-calculate up vector to ensure orthogonality
    up = np.cross(right, front)

    # Construct rotation matrix
    rotation_matrix = np.array([
        right,
        up,
        -front  # Negative front because we want camera to look in that direction.
    ]).T

    # Construct extrinsic matrix
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rotation_matrix
    extrinsic_matrix[:3, 3] = camera_position

    return extrinsic_matrix



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
        #plt.figure(figsize=(10,10))
        #plt.imshow(color_frame)
        #plt.axis('on')
        #plt.show()  
        self.depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        self.color_intrin = color_frame.profile.as_video_stream_profile().intrinsics

        self.depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
        self.colorizer = rs.colorizer()

        self.K = np.array([[self.depth_intrin.fx, 0, self.depth_intrin.ppx],
                           [0, self.depth_intrin.fy, self.depth_intrin.ppy],
                           [0, 0, 1]])

        self.dec_filter = rs.decimation_filter()
        self.spat_filter = rs.spatial_filter()
        self.temp_filter = rs.temporal_filter()
        self.hole_filling_filter = rs.hole_filling_filter()
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
        #filtered = self.dec_filter.process(depth_frame)
        #filtered = self.spat_filter.process(filtered)

        filtered = depth_frame
        #filtered = self.hole_filling_filter.process(filtered)
        #filtered = self.temp_filter.process(filtered)
        #filtered = self.spat_filter.process(filtered)
        return filtered.as_depth_frame()


    def denoise(self, depth_img):
        max_val = np.amax(depth_img)
        min_val = np.amin(depth_img)
        normalized = depth_img - min_val / (max_val - min_val)
        normalized_vis = cv2.normalize(normalized, 0, 255, cv2.NORM_MINMAX)
        idxs = np.where(normalized_vis.ravel() > 0)[0]
        return idxs

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
        depth_image = np.asanyarray(depth_frame.get_data()) # self.colorizer.colorize() 

        #denoised_idxs = self.denoise(depth_image)
        h, w = depth_image.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        u = u.flatten()
        v = v.flatten()

        # Compute 3D points for valid pixels
        points_3d = []
        for u_pixel, v_pixel in zip(u, v):
            point_3d = self.deproject_pixel((u_pixel, v_pixel), depth_frame)
            points_3d.append(np.array(point_3d))
        points_3d = np.array(points_3d).T
        print(points_3d.shape)
        points_3d = np.vstack((points_3d, np.ones([1,points_3d.shape[1]])))
        points_3d = ((np.eye(4).dot(points_3d)).T)[:, 0:3]
        #points_3d = self.deproject(depth_image, self.K)
        colors = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB).reshape(points_3d.shape)/255.
        #points_3d = points_3d[denoised_idxs]
        #colors = colors[denoised_idxs]
        
        return points_3d, colors, color_frame, color_image, depth_frame, depth_image
    
    def seg_to_pc(self, segmap):
        """
        Convert depth and intrinsics to point cloud and optionally point cloud color
        :param depth: hxw depth map in m
        :param K: 3x3 Camera Matrix with intrinsics
        :returns: (Nx3 point cloud, point cloud color)
        """
        align_to = rs.stream.color
        align = rs.align(align_to)
        
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            color_image_grb = np.asanyarray(color_frame.get_data())
            color_image = color_image_grb[:, :, ::-1] # Real sense uses BGR but we want RGB
            
            depth_image = np.asanyarray(depth_frame.get_data())
            
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            
            K = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                        [0, intrinsics.fy, intrinsics.ppy],
                        [0, 0, 1]])

            mask = np.where(depth_image > 0 and segmap != 0)
            x,y = mask[1], mask[0]
            
            normalized_x = (x.astype(np.float32) - K[0,2])
            normalized_y = (y.astype(np.float32) - K[1,2])

            world_x = normalized_x * depth_image[y, x] / K[0,0]
            world_y = normalized_y * depth_image[y, x] / K[1,1]
            world_z = depth_image[y, x]

            pc_rgb = color_image[y,x,:]
                
            pc = np.vstack((world_x, world_y, world_z)).T
            return pc, pc_rgb
    
        except Exception as e:
            print(f"Error getting point cloud: {e}")
            return None

    def stop_stream(self):
        self.pipeline.stop()

    def show_image(self, image):
        cv2.imshow('img', image)
        cv2.waitKey(0)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(f'GPUs: {physical_devices}')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))
import config_utils
from data import regularize_pc_point_count, depth2pc, load_available_input_data

from contact_grasp_estimator import GraspEstimator
#from visualization_utils import visualize_grasps, show_image

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
    #realsense_streamer  = RealsenseStreamer('317222072157')
    realsense_streamer = RealsenseStreamer(serial_no) #317422074281 small
    
    t0 = time.time()
    # Build the model
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
    grasp_estimator.load_weights(sess, saver, FLAGS.ckpt_dir, mode='test')
    t1 = time.time()
    print(f'time taken to load model weights: {round(t1-t0, 2)} seconds')
    os.makedirs('results', exist_ok=True)
    
    seg_cam_window = "Gaze Segmentation (Camera)" # New window for RealSense
    
    cv2.namedWindow(seg_cam_window, cv2.WINDOW_NORMAL) 
    cv2.resizeWindow(seg_cam_window, 640, 480) 
    cv2.moveWindow(seg_cam_window, 1800, 1050)
    
    frames = []
    positions = []
    orientations = []
    while True:
        points_3d, colors, _, realsense_img,  depth_frame, depth_image = realsense_streamer.capture_rgbd()
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pc_full = pcd.points
        pc_full = np.asarray(pc_full)
        print(pc_full.shape)
        #o3d.io.write_point_cloud(f'{serial_no}.pcd', pcd)

        print(str(global_config))
        print('pid: %s'%(str(os.getpid())))

        t0 = time.time()
        #pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(sess, pc_full, pc_segments=pc_segments, 
        #                                                                                  local_regions=local_regions, filter_grasps=filter_grasps, forward_passes=forward_passes)  
        gaze = np.array([230, 343])
        print(realsense_img.shape)
        seg_model = FastSAM('./contact_graspnet/segment/FastSAM-s.pt')
        segmented_cam_img, _ = select_from_sam_everything( #Segments everything and merge masks that is closest to the point prompt
                            seg_model,
                            [gaze],
                            input_img=realsense_img,
                            imgsz=640,
                            iou=0.9,
                            conf=0.4,
                            max_distance=10,#100,
                            device=None,
                            retina=True,
                            #include_largest_mask = True
                        )
        
        print("segmented_cam_img shape:", segmented_cam_img.shape)
        print("segmented_cam_img dtype:", segmented_cam_img.dtype)
        print("segmented_cam_img min/max:", segmented_cam_img.min(), segmented_cam_img.max())
        '''for mask_point in mask_pt_cam:
            transformed_x, transformed_y = mask_point
            cv2.circle(segmented_cam_img, (int(transformed_x), int(transformed_y)), 5, 255, 10)
'''
        cv2.imshow(seg_cam_window, segmented_cam_img.astype(np.uint8) * 255)
        cv2.waitKey(0)
        print('depth shape: ', depth_image.shape)
        #print(depth_image)
        depth_image = depth_image/1000
    
        #pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(sess, pc_full, pc_segments=None, 
        #                                                                                  local_regions=None, filter_grasps=False, forward_passes=1)  
        print('generating grasp from segmented...')
        pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps_from_depth_K_and_2d_seg(sess, depth_image, segmented_cam_img, 
                                                                                                               realsense_streamer.K, local_regions=True,
                                                                                                                 filter_grasps=True)
        
        t1 = time.time()
        print(f'time taken to generate grasps: {round(t1-t0, 2)} seconds')
        #print(pred_grasps_cam)
        extrinsic_matrix = np.load(f"./calib/extrinsic_{serial_no}.npy")
        extrinsic_matrix1 = np.load(f"./calib/extrinsic_{serial_no}_1.npy")
        grasp_width = 0.008
        
        grippers = []
        grippers_pink = []

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)

        #for plotting all grasps
        for i, k in enumerate(pred_grasps_cam):
            if np.any(pred_grasps_cam[k]):
                grasps = pred_grasps_cam[k]
                print(grasps.shape)
                for grasp in grasps:
                    gripper_control_points_closed = grasp_line_plot.copy()
                    gripper_control_points_closed[2:6:2, 1] = np.sign(grasp_line_plot[2:6:2, 1]) * 0.08 / 2  # Change y-axis to simulate opening

                    # Extract rotation and translation from grasp matrix
                    R = np.array(grasp[:3, :3])  # Rotation matrix
                    t = np.array(grasp[:3, 3])   # Translation vector

                    # Apply rotation and translation to gripper points
                    pts = np.matmul(gripper_control_points_closed, R.T)  # Apply rotation
                    pts += t  # Apply translation

                    # Flatten all_pts into a single array
                    all_pts_array = np.vstack([pts])

                    # Create Open3D LineSet
                    line_set = o3d.geometry.LineSet(
                        points=o3d.utility.Vector3dVector(all_pts_array),  # Pass the flattened array
                        lines=o3d.utility.Vector2iVector(connections[0])   # Pass the connections
                    )

                    # Set color for the gripper
                    line_set.paint_uniform_color([0, 0, 0])  # Black color

                    # Add LineSet to the visualizer
                    vis.add_geometry(line_set)

        # Add semantic waypoint (blue sphere)
        semantic_waypoint = realsense_streamer.deproject_pixel(gaze, depth_frame)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)  # Adjust radius as needed
        sphere.translate(semantic_waypoint)  # Translate to the semantic waypoint
        sphere.paint_uniform_color([0, 0, 1])  # Blue color (RGB)
        vis.add_geometry(sphere)
        vis.run()

        # Print message
        print("Getting image of all predicted grasps with gaze point")

        # Set the view using the loaded extrinsic matrix
        view_control = vis.get_view_control()
        param = view_control.convert_to_pinhole_camera_parameters()
        param.extrinsic = extrinsic_matrix
        view_control.convert_from_pinhole_camera_parameters(param)

        # Update visualizer
        vis.update_renderer()
        vis.poll_events()
        time.sleep(1)

        # Capture screen and save as image
        float_buffer = vis.capture_screen_float_buffer()
        float_array = np.asarray(float_buffer)
        image_array = (255.0 * float_array).astype(np.uint8)
        cv2.imwrite("grasps_with_gaze.png", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))


        print("Specifying grasp options")

        distinct_grasps = find_distinct_grasps(pred_grasps_cam, gaze, depth_frame, realsense_streamer, n_grasps=3, max_distance=0.2)
        closest_grasps = find_closest_grasp(pred_grasps_cam, gaze, depth_frame, realsense_streamer)
        grasps = distinct_grasps
        grasps.append(closest_grasps)
        base_link_color = [[1, 0.5, 1], [0.5, 1, 1], [1, 1, 0.5], [0.5, 0.5, 1]]
        for i, grasp in enumerate(grasps):
            gripper = visualize_gripper(grasp, base_link_color[i])
          # gripper_pink = visualize_gripper(grasp, [1, 0.5, 1])
            grippers.append(gripper)
          # grippers_pink.append(gripper_pink)
            ##############################################
            # Simulate gripper opening by adjusting y-axis values
            gripper_control_points_closed = grasp_line_plot.copy()
            gripper_control_points_closed[2:6:2, 1] = np.sign(grasp_line_plot[2:6:2, 1]) * 0.08 / 2  # Change y-axis to simulate opening

            # Extract rotation and translation from grasp matrix
            R = np.array(grasp[:3, :3])  # Rotation matrix
            t = np.array(grasp[:3, 3])   # Translation vector

            # Apply rotation and translation to gripper points
            pts = np.matmul(gripper_control_points_closed, R.T)  # Apply rotation
            pts += t  # Apply translation

            # Flatten all_pts into a single array
            all_pts_array = np.vstack([pts])

            # Create Open3D LineSet
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(all_pts_array),  # Pass the flattened array
                lines=o3d.utility.Vector2iVector(connections[0])   # Pass the connections
            )

            # Set color for the gripper
            line_set.paint_uniform_color(base_link_color[i])  # Red color

            vis.clear_geometries()
            vis.add_geometry(pcd)
            vis.add_geometry(line_set)

            print("Getting image of gripper line set")

            # Set the view using the loaded extrinsic matrix
            view_control = vis.get_view_control()
            param = view_control.convert_to_pinhole_camera_parameters()
            param.extrinsic = extrinsic_matrix
            view_control.convert_from_pinhole_camera_parameters(param)

            vis.update_renderer()
            vis.poll_events()
            time.sleep(1)
            float_buffer = vis.capture_screen_float_buffer()
            float_array = np.asarray(float_buffer)
            image_array = (255.0 * float_array).astype(np.uint8)
            cv2.imwrite(f"grasp_lines_set{i}.png", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))

            # Second camera view
            view_control = vis.get_view_control()
            param = view_control.convert_to_pinhole_camera_parameters()
            param.extrinsic = extrinsic_matrix1
            view_control.convert_from_pinhole_camera_parameters(param)

            vis.update_renderer()
            vis.poll_events()
            time.sleep(1)
            float_buffer = vis.capture_screen_float_buffer()
            float_array = np.asarray(float_buffer)
            image_array = (255.0 * float_array).astype(np.uint8)
            cv2.imwrite(f"grasp_lines_set{i}_1.png", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
            ##############################################
            print("Getting image of coloured gripper")
            vis.clear_geometries()
            vis.add_geometry(pcd)
            for part in gripper:
                vis.add_geometry(part)

            '''R = np.array(grasp[:3, :3])  # Extract rotation
            t = np.array(grasp[:3, 3]) 

            offset_x = 0.2  # Offset along gripper's x-axis
            camera_position = t + R[:, 0] * offset_x  # Camera position
            print("Camera position: ", camera_position)
            lookat = t  # Gripper origin
            front = -R[:, 0]  # Negative x-axis of gripper
            up = -R[:, 2]  # Negative z-axis of gripper (vertical down)

            extrinsic_matrix = get_extrinsic_matrix(camera_position, lookat, front, up)
            print('Calculated extrinsic matrix:', extrinsic_matrix)

            new_extrinsic_matrix = np.eye(4)
            new_extrinsic_matrix[:3, :3] = grasp[:3, :3]
            new_extrinsic_matrix[:3, 3] = grasp[:3, 3]

            T = np.array([
                [0, 0, 1, 0.5],  # Adjusted translation relative to grasp
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 0, 1]
            ])

            new_extrinsic_matrix = new_extrinsic_matrix @ T 
            print('Calculated extrinsic matrix:', new_extrinsic_matrix)
            view_control = vis.get_view_control()
            param = view_control.convert_to_pinhole_camera_parameters()
            param.extrinsic = new_extrinsic_matrix
            view_control.convert_from_pinhole_camera_parameters(param)
            '''
            view_control = vis.get_view_control()
            param = view_control.convert_to_pinhole_camera_parameters()
            param.extrinsic = extrinsic_matrix
            view_control.convert_from_pinhole_camera_parameters(param)

            vis.update_renderer()
            vis.poll_events()
            '''
            vis.run()
            view_control = vis.get_view_control()

            # Get the pinhole camera parameters
            pinhole_camera_parameters = view_control.convert_to_pinhole_camera_parameters()

            # Print the parameters (or save them to a file)
            print("Wanted Camera Parameters:")
            print("  Extrinsic:")
            print(pinhole_camera_parameters.extrinsic)
            '''
            time.sleep(1)
            float_buffer = vis.capture_screen_float_buffer()
            float_array = np.asarray(float_buffer)
            image_array = (255.0 * float_array).astype(np.uint8)
            cv2.imwrite(f"gripper{i}.png", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
            # Second camera view
            view_control = vis.get_view_control()
            param = view_control.convert_to_pinhole_camera_parameters()
            param.extrinsic = extrinsic_matrix1
            view_control.convert_from_pinhole_camera_parameters(param)

            vis.update_renderer()
            vis.poll_events()
            time.sleep(1)
            float_buffer = vis.capture_screen_float_buffer()
            float_array = np.asarray(float_buffer)
            image_array = (255.0 * float_array).astype(np.uint8)
            cv2.imwrite(f"gripper{i}_1.png", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
            ##############################################
            '''vis.clear_geometries()
            vis.add_geometry(pcd)
            for part in gripper_pink:
                vis.add_geometry(part)

            print("Getting image of pink gripper")

            # Set the view using the loaded extrinsic matrix
            view_control = vis.get_view_control()
            param = view_control.convert_to_pinhole_camera_parameters()
            param.extrinsic = extrinsic_matrix
            view_control.convert_from_pinhole_camera_parameters(param)

            vis.update_renderer()
            vis.poll_events()
            time.sleep(1)
            float_buffer = vis.capture_screen_float_buffer()
            float_array = np.asarray(float_buffer)
            image_array = (255.0 * float_array).astype(np.uint8)
            cv2.imwrite(f"gripper_pink{i}.png", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))

            # Second camera view
            view_control = vis.get_view_control()
            param = view_control.convert_to_pinhole_camera_parameters()
            param.extrinsic = extrinsic_matrix1
            view_control.convert_from_pinhole_camera_parameters(param)

            vis.update_renderer()
            vis.poll_events()
            time.sleep(1)
            float_buffer = vis.capture_screen_float_buffer()
            float_array = np.asarray(float_buffer)
            image_array = (255.0 * float_array).astype(np.uint8)
            cv2.imwrite(f"gripper_pink{i}_1.png", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))'''
            ##############################################
            R = np.array(grasp[:3, :3])  # Extract rotation
            t = np.array(grasp[:3, 3])   # Extract translation (position)
            grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.09, origin=t)  # Adjust size as needed
            grasp_frame.rotate(R, center=t) # Apply the grasp rotation
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.translate(t)
            sphere.paint_uniform_color(base_link_color[i])

            # Add to the visualizer
            vis.clear_geometries()
            vis.add_geometry(pcd)
            vis.add_geometry(grasp_frame)
            vis.add_geometry(sphere)

            print("Getting image of gripper axis")

            # Set the view using the loaded extrinsic matrix
            view_control = vis.get_view_control()
            param = view_control.convert_to_pinhole_camera_parameters()
            param.extrinsic = extrinsic_matrix
            view_control.convert_from_pinhole_camera_parameters(param)

            vis.update_renderer()
            vis.poll_events()
            time.sleep(1)
            float_buffer = vis.capture_screen_float_buffer()
            float_array = np.asarray(float_buffer)
            image_array = (255.0 * float_array).astype(np.uint8)
            cv2.imwrite(f"grasp_lines{i}.png", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))

            # Second camera view
            view_control = vis.get_view_control()
            param = view_control.convert_to_pinhole_camera_parameters()
            param.extrinsic = extrinsic_matrix1
            view_control.convert_from_pinhole_camera_parameters(param)

            vis.update_renderer()
            vis.poll_events()
            time.sleep(1)
            float_buffer = vis.capture_screen_float_buffer()
            float_array = np.asarray(float_buffer)
            image_array = (255.0 * float_array).astype(np.uint8)
            cv2.imwrite(f"grasp_lines{i}_1.png", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
            
            ##############################################
            print("Moving to pose")
            position_cam = 1000.0*np.array(grasp[:3, 3])  # Extract translation
            position_rob = np.array(transform(np.array(position_cam).reshape(1,3), TCR))[0]
            rotation_matrix = grasp[:3, :3]
            orientation = transform_rotation_camera_to_robot_roll_yaw_pitch(rotation_matrix, TCR)
            print("Position: ", position_rob)
            print("Orientation: ", orientation)
            positions.append(position_rob)
            orientations.append(orientation)
            robot.move_to_ee_pose(position_rob, orientation)

        #vis.clear_geometries()
        print("Getting image of all the coloured grippers")
        vis.clear_geometries()
        vis.add_geometry(pcd)
        for gripper in grippers:
            for part in gripper:
                vis.add_geometry(part)

        # Set the view using the loaded extrinsic matrix
        view_control = vis.get_view_control()
        param = view_control.convert_to_pinhole_camera_parameters()
        param.extrinsic = extrinsic_matrix
        view_control.convert_from_pinhole_camera_parameters(param)

        vis.update_renderer()
        vis.poll_events()
        time.sleep(1)
        float_buffer = vis.capture_screen_float_buffer()
        float_array = np.asarray(float_buffer)
        image_array = (255.0 * float_array).astype(np.uint8)
        cv2.imwrite(f"all_gripper.png", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
        # second camera view
        view_control = vis.get_view_control()
        param = view_control.convert_to_pinhole_camera_parameters()
        param.extrinsic = extrinsic_matrix1
        view_control.convert_from_pinhole_camera_parameters(param)

        vis.update_renderer()
        vis.poll_events()
        time.sleep(1)
        float_buffer = vis.capture_screen_float_buffer()
        float_array = np.asarray(float_buffer)
        image_array = (255.0 * float_array).astype(np.uint8)
        cv2.imwrite(f"all_gripper_1.png", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
        ##############################################
        '''print("Getting image of all pink grippers")
        vis.clear_geometries()
        vis.add_geometry(pcd)
        for gripper in grippers_pink:
            for part in gripper:
                vis.add_geometry(part)

        # Set the view using the loaded extrinsic matrix
        view_control = vis.get_view_control()
        param = view_control.convert_to_pinhole_camera_parameters()
        param.extrinsic = extrinsic_matrix
        view_control.convert_from_pinhole_camera_parameters(param)

        vis.update_renderer()
        vis.poll_events()
        time.sleep(1)
        float_buffer = vis.capture_screen_float_buffer()
        float_array = np.asarray(float_buffer)
        image_array = (255.0 * float_array).astype(np.uint8)
        cv2.imwrite(f"all_gripper_pink.png", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
        #second camera view
        view_control = vis.get_view_control()
        param = view_control.convert_to_pinhole_camera_parameters()
        param.extrinsic = extrinsic_matrix1
        view_control.convert_from_pinhole_camera_parameters(param)

        vis.update_renderer()
        vis.poll_events()
        time.sleep(1)
        float_buffer = vis.capture_screen_float_buffer()
        float_array = np.asarray(float_buffer)
        image_array = (255.0 * float_array).astype(np.uint8)
        cv2.imwrite(f"all_gripper_pink_1.png", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))'''
        ##############################################
        print("Getting image of all gripper axes")
        vis.clear_geometries()
        vis.add_geometry(pcd)
        for i, grasp in enumerate(grasps):
            R = np.array(grasp[:3, :3])  # Extract rotation
            t = np.array(grasp[:3, 3])   # Extract translation (position)
            grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=t)  # Adjust size as needed
            grasp_frame.rotate(R, center=t) # Apply the grasp rotation
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.translate(t)
            sphere.paint_uniform_color(base_link_color[i])
            vis.add_geometry(grasp_frame)
            vis.add_geometry(sphere)
           #vis.add_geometry(label)

        # Set the view using the loaded extrinsic matrix
        view_control = vis.get_view_control()
        param = view_control.convert_to_pinhole_camera_parameters()
        param.extrinsic = extrinsic_matrix
        view_control.convert_from_pinhole_camera_parameters(param)

        vis.update_renderer()
        vis.poll_events()
        time.sleep(1)
        float_buffer = vis.capture_screen_float_buffer()
        float_array = np.asarray(float_buffer)
        image_array = (255.0 * float_array).astype(np.uint8)
        cv2.imwrite(f"all_grasp_lines.png", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
        #second camera view
        view_control = vis.get_view_control()
        param = view_control.convert_to_pinhole_camera_parameters()
        param.extrinsic = extrinsic_matrix1
        view_control.convert_from_pinhole_camera_parameters(param)

        vis.update_renderer()
        vis.poll_events()
        time.sleep(1)
        float_buffer = vis.capture_screen_float_buffer()
        float_array = np.asarray(float_buffer)
        image_array = (255.0 * float_array).astype(np.uint8)
        cv2.imwrite(f"all_grasp_lines_1.png", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
        ##############################################
        print("Getting image of all gripper_line_sets")
        vis.clear_geometries()
        vis.add_geometry(pcd)
        for i, grasp in enumerate(grasps):
            gripper_control_points_closed = grasp_line_plot.copy()
            gripper_control_points_closed[2:6:2, 1] = np.sign(grasp_line_plot[2:6:2, 1]) * 0.08 / 2  # Change y-axis to simulate opening

            # Extract rotation and translation from grasp matrix
            R = np.array(grasp[:3, :3])  # Rotation matrix
            t = np.array(grasp[:3, 3])   # Translation vector

            # Apply rotation and translation to gripper points
            pts = np.matmul(gripper_control_points_closed, R.T)  # Apply rotation
            pts += t  # Apply translation

            # Flatten all_pts into a single array
            all_pts_array = np.vstack([pts])

            # Create Open3D LineSet
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(all_pts_array),  # Pass the flattened array
                lines=o3d.utility.Vector2iVector(connections[0])   # Pass the connections
            )

            # Set color for the gripper
            line_set.paint_uniform_color(base_link_color[i]) 
            # Add to the visualizer
            vis.add_geometry(line_set)

        # Set the view using the loaded extrinsic matrix
        view_control = vis.get_view_control()
        param = view_control.convert_to_pinhole_camera_parameters()
        param.extrinsic = extrinsic_matrix
        view_control.convert_from_pinhole_camera_parameters(param)

        vis.update_renderer()
        vis.poll_events()
        time.sleep(1)
        float_buffer = vis.capture_screen_float_buffer()
        float_array = np.asarray(float_buffer)
        image_array = (255.0 * float_array).astype(np.uint8)
        cv2.imwrite(f"all_grasp_lines_sets.png", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
        #second camera view
        view_control = vis.get_view_control()
        param = view_control.convert_to_pinhole_camera_parameters()
        param.extrinsic = extrinsic_matrix1
        view_control.convert_from_pinhole_camera_parameters(param)

        vis.update_renderer()
        vis.poll_events()
        time.sleep(1)
        float_buffer = vis.capture_screen_float_buffer()
        float_array = np.asarray(float_buffer)
        image_array = (255.0 * float_array).astype(np.uint8)
        cv2.imwrite(f"all_grasp_lines_sets_1.png", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
        vis.destroy_window()
        ##############################################
        data = np.array(list(zip(grasps, positions, orientations)), dtype=object)
        np.save("grasp_data.npy", data)

        sys.exit()

        



