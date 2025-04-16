import cv2
import numpy as np
import pyrealsense2 as rs
from rs_streamer import RealsenseStreamer
from collections import deque

class HomographyManager:
    def __init__(self, serial_no):
        self.serial_no = serial_no
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.homography_matrix = None
        self.realsense_streamer = None
        self.pipeline = None
        self.depth_frame = None
        self.matches_history = deque(maxlen=10)
        self.previous_homography = None
        self.alpha = 0.01  # smoothing factor, lower for more stability

    def start_camera(self):
        self.realsense_streamer = RealsenseStreamer(self.serial_no)
        self.pipeline = self.realsense_streamer.pipeline
        print("RealSense camera started")

    def detect_and_compute(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(self, des1, des2):
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
        return good_matches

    def compute_homography_from_matches(self, kp1, kp2, matches):
        if len(matches) < 4:
            return None
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)
        return H

    def smooth_homography(self, H_new):
        if H_new is None:
            return self.previous_homography
        if self.previous_homography is None:
            self.previous_homography = H_new
            return H_new
        H_smoothed = self.alpha * H_new + (1 - self.alpha) * self.previous_homography
        H_smoothed /= H_smoothed[2, 2]
        self.previous_homography = H_smoothed
        return H_smoothed

    def apply_homography(self, gaze_coordinates):
        if self.homography_matrix is None or gaze_coordinates is None:
            return None, None
        point = np.array([gaze_coordinates[0], gaze_coordinates[1], 1.0])
        transformed = self.homography_matrix @ point
        transformed /= transformed[2]
        return transformed[0], transformed[1]

    def process_frame(self, aria_image, gaze_coordinates, show_matches=True):
        frames = self.pipeline.wait_for_frames()
        for _ in range(2):
            _, color_image, depth_frame, _ = self.realsense_streamer.capture_rgbd()
        self.depth_frame = depth_frame

        kp_aria, des_aria = self.detect_and_compute(aria_image)
        kp_rs, des_rs = self.detect_and_compute(color_image)

        if des_aria is None or des_rs is None:
            return color_image, None, None

        matches = self.match_features(des_aria, des_rs)
        if len(matches) >= 4:
            self.matches_history.append((kp_aria, kp_rs, matches))

        # Aggregate matches for smoothing
        all_src_pts = []
        all_dst_pts = []
        for kp_a, kp_r, m in self.matches_history:
            for match in m:
                all_src_pts.append(kp_a[match.queryIdx].pt)
                all_dst_pts.append(kp_r[match.trainIdx].pt)

        if len(all_src_pts) >= 4:
            H_new, _ = cv2.findHomography(np.float32(all_src_pts), np.float32(all_dst_pts), cv2.RANSAC, 4.0)
            self.homography_matrix = self.smooth_homography(H_new)

        # Optional: visualize feature matches
        if show_matches:
            match_img = cv2.drawMatches(
                aria_image, kp_aria,
                color_image, kp_rs,
                matches, None,
                matchColor=(0, 255, 0),
                singlePointColor=(255, 0, 255),
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            cv2.imshow("Feature Matches", match_img)
            cv2.waitKey(1)

        transformed_x, transformed_y = self.apply_homography(gaze_coordinates)
        return color_image, transformed_x, transformed_y
