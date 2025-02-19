import cv2
import numpy as np
import time
import pyrealsense2 as rs
from rs_streamer import RealsenseStreamer
from SAMInference import run_fastsam_inference

def get_connected_devices():
    context = rs.context()
    devices = []
    for d in context.devices:
        devices.append(d.get_info(rs.camera_info.serial_number))
    return devices

class HomographyManager:
    def __init__(self, serial_no = '317422075456'):
        
        self.serial_no = serial_no
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.homography_matrix = None

    def start_camera(self):
        # devices = get_connected_devices()
        # if not devices:
        #     raise Exception("No Realsense device connected")
        # camConfig = rs.config()
        # camConfig.enable_device(devices[0])
        # pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        # pipeline_profile = camConfig.resolve(pipeline_wrapper)
        # device = pipeline_profile.get_device()
        # found_rgb = False
        # for s in device.sensors:
        #     if s.get_info(rs.camera_info.name) == 'RGB Camera':
        #         found_rgb = True
        #         break
        # if not found_rgb:
        #     raise Exception("The demo requires Depth camera with Color sensor")

        # camConfig.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # self.pipeline.start(camConfig)
        self.realsense_streamer = RealsenseStreamer(self.serial_no)
        self.pipeline = self.realsense_streamer.pipeline
        self.config = self.realsense_streamer.config
        print("Realsense camera started")

    def get_point_cloud(self, gaze_coordinates):
        """
        Returns a 3D point cloud only with the 25cm x 25cm x 25cm volume around the gaze coordinates.

        Returns:
            vtx (numpy.ndarray): Nx3 array containing (X, Y, Z) coordinates of the point cloud.
        """
        
        # Align depth to color
        align_to = rs.stream.color
        align = rs.align(align_to)
        
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            depth_image = np.asanyarray(depth_frame.get_data())
            
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            
            K = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                        [0, intrinsics.fy, intrinsics.ppy],
                        [0, 0, 1]])
            
            if not depth_frame or not color_frame:
                print("Failed to capture frames.")
                return None

            start_time = time.time()

            #1, 2, 3 - Ensure Gaze coordinates are entered correctly for array indexing
            if len(gaze_coordinates) != 2:
                    raise ValueError(
                        "gaze_coordinates must be a tuple or list of length 2 (y, x)"
                    )

            #1, 2, 3 - Swap if necessary
            gaze_y, gaze_x = gaze_coordinates

            depth_value = depth_image[gaze_y, gaze_x] / 1000 # assume millimeters
            semantic_waypoint = rs.rs2_deproject_pixel_to_point(intrinsics, [gaze_x, gaze_y], depth_value) # order is intentional here, rs2_deproject_pixel_to_point expects [x,y]
            #print(f"Semantic Waypoint (x, y, z): {semantic_waypoint}")

            # 4 - Optimize: Generate a Bounding volume
            x_low = semantic_waypoint[0] - 0.125
            x_high = semantic_waypoint[0] + 0.125
            y_low = semantic_waypoint[1] - 0.125
            y_high = semantic_waypoint[1] + 0.125

            # Optimized: create point cloud and quickly filter to array. If we want an accurate extraction in space, we have to extract the entire point cloud and use it to filter

            pc = rs.pointcloud()
            points_obj = pc.calculate(depth_frame)

            vtx = np.asanyarray(points_obj.get_vertices()).view(np.float32).reshape(-1, 3)

            # - boolean masking for filtering.

            mask = (
                (vtx[:, 0] > x_low)
                & (vtx[:, 0] < x_high)
                & (vtx[:, 1] > y_low)
                & (vtx[:, 1] < y_high)
            )

            # use mask for filtering
            vtx_filtered = vtx[mask]

            indices = np.random.choice(len(vtx_filtered), len(vtx_filtered)*0.25, replace=False)
            points =  points[indices]

            end_time = time.time()

            # Log total processing time
            print(f"get_point_cloud execution time: {end_time - start_time} seconds")

            np.savez("3dpoint_data.npz", K=K, xyz=vtx_filtered)
            print("3D point cloud data saved to 3dpoint_data.npz")

            return vtx_filtered
        
        except Exception as e:
            print(f"Error getting point cloud: {e}")
            return None

            
    def fetch_3D_scene(self, point_prompt = "[[330,330]]"):
        for _ in range(2):
            t_start = time.time()

            # Get frames
            frames = self.pipeline.wait_for_frames()
            align = rs.align(rs.stream.color)
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            #_, color_image, depth_frame, depth_img = self.realsense_streamer.capture_rgbd()
            #color_image can be directly obtained here 
            #the depth_frame here is in RGB  as compared to the depth image
            #depth_img is the same as depth_image obtained from the depth_frame
            if not depth_frame or not color_frame:
                print("Failed to get frames")
                continue

            # Get camera intrinsics
            depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

            # Convert depth frame to numpy array (in meters)
            depth_image = np.asanyarray(depth_frame.get_data()) / 1000.0  # Convert to meters
            color_image_grb = np.asanyarray(color_frame.get_data())
            color_image = color_image_grb[:, :, ::-1] # Real sense uses BGR but we want RGB
            # Get camera matrix K
            K = np.array([[depth_intrinsics.fx, 0, depth_intrinsics.ppx],
                        [0, depth_intrinsics.fy, depth_intrinsics.ppy],
                        [0, 0, 1]])
            
            ann = run_fastsam_inference(
                        model_path="./FastSAM-s.pt",
                        input_img=color_image,
                        imgsz=1024,
                        iou=0.9,
                        conf=0.4,
                        point_prompt=point_prompt,
                        point_label="[1,0]",
                        better_quality=False,
                        device=None,
                        retina=True,
                        withContours=False,
                    )

            segmented = ann.astype(np.int32)
            np.savez('depth_data.npz', depth=depth_image, K=K, segmap =segmented, color=color_image)
            print("Depth data saved to depth_data.npz")

            # Generate Point Cloud
            pc = rs.pointcloud()
            pc.map_to(color_frame)
            points = pc.calculate(depth_frame)

            # Convert point cloud to numpy array
            vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

            t_end = time.time()
            print(f'Vertices captured in {round(t_end - t_start, 2)} seconds')
            print(vertices.shape)

            time.sleep(1)

        print(vertices)
        return vertices

        

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
            _, color_image, depth_frame, depth_img = self.realsense_streamer.capture_rgbd()
        self.depth_frame = depth_frame
        cam_corners, cam_ids = self.detect_aruco_markers(color_image)

        if cam_ids is not None:
            cv2.aruco.drawDetectedMarkers(color_image, cam_corners, cam_ids)

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
    
    
    
