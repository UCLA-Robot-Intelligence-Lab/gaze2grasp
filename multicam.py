import sys
import pyrealsense2 as rs
import random
import pprint
import numpy as np
import cv2
import copy
from cv2 import aruco
import imageio
from rs_streamer import RealsenseStreamer, MarkSearch
import torch
import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import time
from contact_graspnet.pc_utils import merge_pcls
from rs_streamer import RealsenseStreamer, MarkSearch
from contact_graspnet.calib_utils.solver import Solver
from contact_graspnet.calib_utils.linalg_utils import transform

sys.path.append(os.path.join(os.path.dirname(__file__), '/home/u-ril/URIL/xArm-Python-SDK'))
from xarm import XArmAPI

from dataclasses import dataclass
#from vision_utils.pc_utils import deproject, deproject_pixels, project, transform_points, draw_registration_result, rescale_pcd, align_pcds, merge_pcls, denoise
#from vision_utils.pc_utils import *

GRIPPER_SPEED, GRIPPER_FORCE, GRIPPER_MAX_WIDTH, GRIPPER_TOLERANCE = 0.1, 40, 0.08570, 0.01

def is_position_in_range(position, x_range=(60, 650), y_range=(-650, 650), z_range=(5, 600)):
    """Check if position is within valid workspace ranges."""
    x, y, z = position
    return (x_range[0] <= x <= x_range[1] and
            y_range[0] <= y <= y_range[1] and
            z_range[0] <= z <= z_range[1])

ip = '192.168.1.223'

@dataclass
class XArmConfig:
    """
    Configuration class for some (not all!) xArm7/control parameters. The important ones are here.
    You can or should change most of these to your liking, potentially with the exception of tcp_maxacc
    
    :config_param tcp_maxacc: TCP (Tool Center Point, i.e., end effector) maximum acceleration
    :config_param position_gain: Increasing this value makes the position gain increase
    :config_param orientation_gain: Increasing this value makes the orientation gain increase
    :config_param alpha: This is a pseudo-smoothing factor
    :config_param control_loop_rate: Self-descriptive
    :config verbose: Helpful debugging / checking print steps
    """
    tcp_maxacc: int = 5000
    position_gain: float = 10.0
    orientation_gain: float = 10.0
    alpha: float = 0.5
    control_loop_rate: int = 50
    verbose: bool = True


class XarmEnv:
    def __init__(self):

    
        xarm_cfg = XArmConfig()

        self.arm = XArmAPI(ip)
        self.arm.connect()

        # This may be unsafe
        self.arm.clean_error()
        self.arm.clean_warn()

        # Robot arm below:
        ret = self.arm.motion_enable(enable=True)
        if ret != 0:
            print(f"Error in motion_enable: {ret}")
            sys.exit(1)
        ret = self.arm.set_gripper_enable(True)
        if ret != 0:
            print(f"Error in gripper_enable: {ret}")
            sys.exit(1)
            
        self.arm.set_tcp_maxacc(xarm_cfg.tcp_maxacc)  
        self.arm.set_mode(0)
        self.arm.set_state(state=0)     

        ret = self.arm.set_mode(1) # This sets the mode to serve motion mode
        if ret != 0:
            print(f"Error in set_mode: {ret}")
            sys.exit(1)

        ret = self.arm.set_state(0) # This sets the state to sport (ready) state
        if ret != 0:
            print(f"Error in set_state: {ret}")
            sys.exit(1)
        
        ret, state = self.arm.get_state()
        if ret != 0:
            print(f"Error getting robot state: {ret}")
            sys.exit(1)
        
        if state != 0:
            print(f"Robot is not ready to move. Current state: {state}")
            sys.exit(1)
        else:
            print(f"Robot is ready to move. Current state: {state}")
        
        self.go_home()
        self.grasp(None)

    def grasp(self, grasp):
        gripper_open = 800
        gripper_closed = 10
        if grasp == None:
            ret = self.arm.set_gripper_position(gripper_open, wait=False)
            if ret != 0:
                print(f"Error in set_gripper_position (close): {ret}")
        else:
            grasp = min(grasp, gripper_open)
            grasp = max(grasp, gripper_closed)
            ret = self.arm.set_gripper_position(grasp, wait=False)
            if ret != 0:
                print(f"Error in set_gripper_position (open): {ret}")

    def go_home(self):
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        self.arm.set_servo_angle(angle=[0, 0, 0, 105, 0, 105, 0], speed=50, wait=True)
        
    def move_to_ee_pose(self,position,orietation,within_bounds = True):
        if is_position_in_range(position):
            ret = self.arm.set_position(x=position[0], y=position[1], z=position[2], roll=orietation[0], pitch=orietation[1], yaw=orietation[2], speed=200, is_radian=False, wait=True)
            return ret
        else:
            print(f"Target position {position} is out of range")
            return None
        
class MultiCam:
    def __init__(self, serial_nos):
        self.cameras = []
        for serial_no in serial_nos:
            self.cameras.append(RealsenseStreamer(serial_no))

        self.transforms = None

        if os.path.exists('calib/transforms.npy'):
            self.transforms = np.load('calib/transforms.npy', allow_pickle=True).item()

        if os.path.exists('calib/icp_tf.npy'):
            self.icp_tf = np.load('calib/icp_tf.npy', allow_pickle=True).item()
            #self.icp_tf = None
        else:
            self.icp_tf = None

    def crop(self, pcd, min_bound=[0.2,-0.35,0.10], max_bound=[0.9, 0.3, 0.5]): 
        idxs = np.logical_and(np.logical_and(
                      np.logical_and(pcd[:,0] > min_bound[0], pcd[:,0] < max_bound[0]),
                      np.logical_and(pcd[:,1] > min_bound[1], pcd[:,1] < max_bound[1])),
                      np.logical_and(pcd[:,2] > min_bound[2], pcd[:,2] < max_bound[2]))
        return idxs

    def take_rgbd_mm(self, visualize=True):
        rgb_images = {cam.serial_no: None for cam in self.cameras}
        depth_images = {cam.serial_no: None for cam in self.cameras}

        merged_points = []
        merged_colors = []

        icp_tfs = []
        cam_ids = []

        for idx, cam in enumerate(self.cameras):
            _, rgb_image, depth_frame, depth_img_vis = cam.capture_rgbd()
            depth_img = np.asarray(depth_frame.get_data())
            # Load transformation for the camera
            tf = self.transforms[cam.serial_no]['tcr']

            # Generate pixel coordinates
            h, w = depth_img.shape
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            u = u.flatten()
            v = v.flatten()

            # Compute 3D points for valid pixels
            points_3d = []
            for u_pixel, v_pixel in zip(u, v):
                point_3d = cam.deproject((u_pixel, v_pixel), depth_frame)
                points_3d.append(np.array(point_3d)*1000) 
            points_3d = np.array(points_3d)

            # Apply transformation to 3D points
            points_3d_homog = np.vstack((points_3d.T, np.ones((1, points_3d.shape[0]))))  # Shape: (4, N)
            points_3d_transformed = (tf @ points_3d_homog).T 
            # Convert colors and normalize
            colors = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0

            # Append transformed points and colors
            merged_points.append(points_3d_transformed)
            merged_colors.append(colors)

            # Visualize individual point cloud (optional)
            if visualize:
                o3d_pcd = o3d.geometry.PointCloud()
                o3d_pcd.points = o3d.utility.Vector3dVector(points_3d_transformed)
                o3d_pcd.colors = o3d.utility.Vector3dVector(colors)
                o3d.visualization.draw_geometries([o3d_pcd])
                o3d.io.write_point_cloud(f"pc_{idx}.pcd", o3d_pcd)

            if idx > 0:
                cam_ids.append(cam.serial_no)
                if self.icp_tf is not None:
                    icp_tfs.append(self.icp_tf[cam.serial_no])
        
        # Merge point clouds
        pcd_merged = merge_pcls(merged_points, merged_colors, tfs=icp_tfs, cam_ids=cam_ids, visualize=visualize)
        return rgb_images, depth_images, pcd_merged

    def calibrate_cam_mm(self, robot=None):
        if not os.path.exists('calib'):
            os.mkdir('calib')
            curr_calib = {}
        else:
            curr_calib = np.load('calib/transforms.npy', allow_pickle=True).item()
        self.marker_search = MarkSearch()

        self.solver = Solver()
        
        def gen_calib_waypoints(start_pos):
            waypoints = []
            for i in np.linspace(300,500,4):
                for j in np.linspace(-250,250,4):
                    for k in np.linspace(100,250,3):
                        waypoints.append(np.array([i,j,k]))
            return waypoints
    

        if robot is None:
            robot = XarmEnv()

        # Get ee pose
        ee_pos, ee_euler = robot.pose_ee()
        print('ee_euler',ee_euler)
        #ee_euler = [-180,0,0]
        ee_euler = [169, 26, 27] #81
        #ee_euler = [174,-27,35]#56

        waypoints = gen_calib_waypoints(ee_pos)

        calib_eulers = []
        z_offsets = [0, 0]
        for z_off in z_offsets:
            calib_euler = ee_euler 
            calib_eulers.append(calib_euler)

        waypoints_rob = []
        waypoints_cam = {c.serial_no:[] for c in self.cameras}

        state_log = robot.move_to_ee_pose(
            ee_pos, calib_euler
        )

        itr = 0
        for i, waypoint in enumerate(waypoints):
            print(itr, waypoint)
            itr+=1
            successful_waypoint = True

            intermed_waypoints = {}
            for idx, cam in enumerate(self.cameras):
                calib_euler = calib_eulers[idx]
                state_log = robot.move_to_ee_pose(
                    waypoint, calib_euler,
                )
                _, rgb_image, depth_frame, depth_img = cam.capture_rgbd()
                (u,v), vis = self.marker_search.find_marker(rgb_image)
                    
                print("Waypoint found: ", u, v)
                if u is None:
                    successful_waypoint = False
                    break
                waypoint_cam = np.array(cam.deproject((u,v), depth_frame))
                waypoint_cam = 1000.0*waypoint_cam
                print('waypoint_cam', waypoint_cam)
                intermed_waypoints[cam.serial_no] = waypoint_cam
            
            if successful_waypoint:
                rot = R.from_euler('ZYX', ee_euler, degrees=True)
                waypoint_rob = waypoint - 152 *rot.as_matrix()[:, 2]
                waypoints_rob.append(waypoint_rob)
                for k in intermed_waypoints:
                    waypoints_cam[k].append(intermed_waypoints[k])

        pprint.pprint(waypoints_cam)
        pprint.pprint(waypoints_rob)

        transforms = {}

        waypoints_rob = np.array(waypoints_rob)

        for cam in self.cameras:
            waypoints_cam_curr = waypoints_cam[cam.serial_no]
            waypoints_cam_curr = np.array(waypoints_cam_curr)
            trc, tcr = self.solver.solve_transforms(waypoints_rob, waypoints_cam_curr)
            transforms[cam.serial_no] = {'trc': trc, 'tcr':tcr}
            

        curr_calib.update(transforms)
        np.save('calib/transforms.npy', curr_calib)

class PixelSelector:
    def __init__(self):
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
            cv2.circle(self.img, (x, y), 3, (255, 255, 0), -1)

    def run(self, img):
        self.load_image(img)
        self.clicks = []
        cv2.namedWindow('pixel_selector')
        cv2.setMouseCallback('pixel_selector', self.mouse_callback)
        while True:
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
        return self.clicks

def goto(robot, realsense_streamer, pixel_selector, TCR, refine=False):
    # right
    
    print(TCR)
    """
    UPDATE THE TCR HERE TO GO TO THE DESIRED POSE
    """
    #for 56
    #TCR[0,3] += 75#x
    #TCR[1,3] += 75#y
    #TCR[2,3] += 40#z
    #for 81
    #TCR[0,3] += -75#x
    #TCR[1,3] += -30#y
    #TCR[2,3] += 20#z

    for i in range(5):
        _, rgb_image, depth_frame, depth_img = realsense_streamer.capture_rgbd()

    pixels = pixel_selector.run(rgb_image)
    waypoint_cam = 1000*np.array(realsense_streamer.deproject(pixels[0], depth_frame))
    print(waypoint_cam)
    waypoint_rob = transform(np.array(waypoint_cam).reshape(1,3), TCR)

    # Get waypoints in robot frame
    ee_pos_desired = np.array(waypoint_rob)[0]+np.array([0,0,152])
    print('ee_pos_desired', ee_pos_desired)

    # Put robot in canonical orientation
    robot.go_home()
    ee_pos, ee_euler = robot.pose_ee()

    state_log = robot.move_to_ee_pose(
        ee_pos, ee_euler, 
    )
    state_log = robot.move_to_ee_pose(
        ee_pos_desired, ee_euler, 
    )
    return TCR

   
if __name__ == "__main__":
    """
    CALIBRATION
    - Change the euler angles and waypoints accordingly if required for each camera
    """
    '''
    multi_cam = MultiCam(['317422074281']) 
    multi_cam.calibrate_cam_mm()
    
    multi_cam = MultiCam(['317422075456']) 
    multi_cam.calibrate_cam_mm()
    '''

    """
    FINE TUNING CALIBRATION
    - Update goto function to go to the desired pose
    """
    '''
    serial_no = '317422074281'
    #serial_no = '317422075456'
    realsense_streamer = RealsenseStreamer(serial_no)
    transforms = np.load(f'calib/transforms.npy', allow_pickle=True).item()
    TCR = transforms[serial_no]['tcr']
    robot = XarmEnv()
    pixel_selector = PixelSelector()
    TCR = goto(robot, realsense_streamer, pixel_selector, TCR, refine=True)
    transforms[serial_no]['tcr'] = TCR

    res = input('save?')
    if res == 'y' or res == 'Y':
        np.save('calib/transforms.npy', transforms)
        print('SAVED')
    '''

    """
    TESTING THAT BOTH CAMERAS ARE CORRECTLY CALIBRATED TO THE SAME ROBOT BASE
    """
    # Uncomment to take an image + merged point cloud
    multi_cam = MultiCam(['317422075456' , '317422074281']) 
    rgb_images, depth_images, pcd_merged = multi_cam.take_rgbd_mm()
    