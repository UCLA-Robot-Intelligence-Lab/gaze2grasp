import cv2
import numpy as np
import pyrealsense2 as rs
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))

from rs_streamer import RealsenseStreamer

SERIAL_NO_81 = '317422074281'  # Camera serial number
SERIAL_NO_56 = '317422075456'

def get_connected_devices():
    context = rs.context()
    devices = []
    for d in context.devices:
        devices.append(d.get_info(rs.camera_info.serial_number))
    return devices

class HomographyManager:
    def __init__(self, serial_no):
        
        self.serial_no = serial_no
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.homography_matrix = None

    def start_camera(self):
        self.realsense_streamer = RealsenseStreamer(self.serial_no)
        self.pipeline = self.realsense_streamer.pipeline
        print("Realsense camera started")


    def detect_aruco_markers(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        corners, ids, rejected = detector.detectMarkers(gray_image)
        return corners, ids

    def match_aruco_markers(self, corners_aria, ids_aria, corners_realsense, ids_realsense):
        matched_points_aria = []
        matched_points_realsense = []

        if ids_aria is not None and ids_realsense is not None:
            for i, id_aria in enumerate(ids_aria):
                if id_aria in ids_realsense:
                    idx_realsense = list(ids_realsense).index(id_aria)

                    corners_aria_marker = corners_aria[i]
                    corners_realsense_marker = corners_realsense[idx_realsense]

                    center_aria = np.mean(corners_aria_marker, axis=0)
                    center_realsense = np.mean(corners_realsense_marker, axis=0)

                    matched_points_aria.append(center_aria)
                    matched_points_realsense.append(center_realsense)

        return np.array(matched_points_aria), np.array(matched_points_realsense)

    def compute_homography(self, points_aria, points_realsense):
        if len(points_aria) < 4 or len(points_realsense) < 4:
            return None
        points_aria = np.array(points_aria, dtype="float32")
        points_realsense = np.array(points_realsense, dtype="float32")
        homography_matrix, _ = cv2.findHomography(points_aria, points_realsense)
        return homography_matrix

    def apply_homography(self, gaze_coordinates):
        if self.homography_matrix is None or gaze_coordinates is None:
            return None, None

        gaze_homogeneous = np.array([gaze_coordinates[0], gaze_coordinates[1], 1.0])
        transformed_coordinates = np.dot(self.homography_matrix, gaze_homogeneous)
        transformed_coordinates /= transformed_coordinates[2]
        transformed_x, transformed_y = transformed_coordinates[0], transformed_coordinates[1]
        return transformed_x, transformed_y

    def process_frame(self, aria_corners, aria_ids, gaze_coordinates):
        frames = self.pipeline.wait_for_frames()
         # print('waiting for depth data...')
        for i in range(2):
            #points_3d, pcd_colors, _, color_image, depth_frame, depth_img = self.realsense_streamer.capture_rgbdpc()
            _, color_image, depth_frame, depth_img = self.realsense_streamer.capture_rgbd()
        self.depth_frame = depth_frame
        cam_corners, cam_ids = self.detect_aruco_markers(color_image)

        # if cam_ids is not None:
        #     cv2.aruco.drawDetectedMarkers(color_image, cam_corners, cam_ids)

        matched_points_aria, matched_points_cam = self.match_aruco_markers(
            aria_corners, aria_ids, cam_corners, cam_ids
        )
        matched_points_aria = matched_points_aria.reshape(-1,2) if len(matched_points_aria) > 0 else np.array([])
        matched_points_cam = matched_points_cam.reshape(-1,2) if len(matched_points_cam) > 0 else np.array([])
        points_aria = np.array(matched_points_aria, dtype=np.float32)
        points_realsense = np.array(matched_points_cam, dtype=np.float32)

        if len(matched_points_aria) >= 4:
            self.homography_matrix = self.compute_homography(points_aria, points_realsense)

        transformed_x, transformed_y = self.apply_homography(gaze_coordinates)

        return color_image, transformed_x, transformed_y
    
    # def get_vector(self, x, y):
    #     if self.depth_frame is None:
    #         print("Depth frame is not available.")
    #         return None
        
    #     x = int(x)
    #     y = int(y)
        
    #     depth = self.depth_frame.get_distance(x, y)
    #     if depth == 0:
    #         print("Depth is zero.")
    #         return None
        
    #     depth_intrinsics = self.depth_frame.profile.as_video_stream_profile().get_intrinsics()

    #     point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)

    #     return point_3d
    
    # def projected_point_test(self, x):
    #     x_3d, y_3d, z_3d = x
                            
    #     # Draw a line from the camera origin to the gaze point (3D vector)
    #     origin = (0, 0, 0)  # Camera center
    #     gaze_point = (x_3d, y_3d, z_3d)
    #     depth_intrinsics = self.depth_frame.profile.as_video_stream_profile().get_intrinsics()
    #     # If you want to visualize it in 2D, you need to project the 3D point to 2D (for rendering on the image)
    #     projected_point = rs.rs2_project_point_to_pixel(depth_intrinsics, gaze_point)

    #     return projected_point