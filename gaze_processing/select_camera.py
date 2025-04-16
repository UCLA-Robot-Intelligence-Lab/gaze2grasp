import cv2
import open3d as o3d
import numpy as np
from rs_streamer import RealsenseStreamer
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import KMeans
from visualizations.live_visualization import visualize_gripper_with_cylinders



# Run in terminal using:
# python -m gaze_processing.select_camera

def cluster_normals_with_position(points, normals, n_clusters=4, direction_weight=0.5):
    points = np.asarray(points)
    normals = np.asarray(normals)

    # Normalize normals just in case
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    # Normalize positions for fair clustering (optional but helpful)
    pos_min = points.min(axis=0)
    pos_max = points.max(axis=0)
    norm_positions = (points - pos_min) / (pos_max - pos_min + 1e-8)

    # Combine features
    features = np.hstack([
        norm_positions * (1 - direction_weight),
        normals * direction_weight
    ])

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(features)

    return labels

def get_cluster_centroids(points, normals, labels):
    points = np.asarray(points)
    normals = np.asarray(normals)
    labels = np.asarray(labels)

    cluster_centroids = []

    for cluster_id in np.unique(labels):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_points = points[cluster_indices]
        cluster_normals = normals[cluster_indices]

        # Compute mean of positions and normals
        mean_point = cluster_points.mean(axis=0)
        mean_normal = cluster_normals.mean(axis=0)
        mean_normal /= np.linalg.norm(mean_normal)  # Re-normalize

        cluster_centroids.append([mean_point, mean_normal])

    return cluster_centroids


def normal_to_pose_matrix(point, normal, pitch_degrees=0):
    """
    Converts a surface normal and point into a 4x4 pose matrix.

    Args:
        point: 3D coordinates of the point.
        normal: 3D unit normal vector.
        pitch_degrees: Optional pitch angle in degrees around x-axis of the frame.

    Returns:
        4x4 transformation matrix (numpy array).
    """
    normal = np.array(normal)
    point = np.array(point)
    normal = normal / np.linalg.norm(normal)

    z_axis = normal

    # Arbitrary vector not parallel to z
    arbitrary = np.array([0, 0, 1]) if abs(z_axis[2]) < 0.99 else np.array([1, 0, 0])

    x_axis = np.cross(arbitrary, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # Rotation matrix (before pitch)
    rot_matrix = np.stack([x_axis, y_axis, z_axis], axis=1)

    # Apply pitch rotation around x-axis
    if pitch_degrees != 0:
        pitch_rad = np.deg2rad(pitch_degrees)
        pitch_rot = R.from_euler('x', pitch_rad).as_matrix()
        rot_matrix = rot_matrix @ pitch_rot

    # Build 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = rot_matrix
    T[:3, 3] = point
    return T


SERIAL_NO_81 = '317422074281'  # Camera serial number
SERIAL_NO_56 = '317422075456'
# Load calibration data
transforms = np.load('calib/transforms.npy', allow_pickle=True).item()
TCR_81 = transforms[SERIAL_NO_81]['tcr']
TCR_56 = transforms[SERIAL_NO_56]['tcr']
TCR_81[:3, 3] /= 1000.0
TCR_56[:3, 3] /= 1000.0

TCR = [TCR_81, TCR_56]

realsense_streamer_81 = RealsenseStreamer(SERIAL_NO_81)
realsense_streamer_56 = RealsenseStreamer(SERIAL_NO_56)

def capture_and_process_rgbd(realsense_streamer):
    points_3d, colors, _, realsense_img, depth_frame, depth_image = realsense_streamer.capture_rgbdpc()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd, realsense_img, depth_frame, depth_image

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


def select_viewpts(points, streamers, depth_frames, realsense_imgs,TCR, threshold=0.05):
    """
    Returns True if the distance between point1 and point2 is greater than threshold.
    """
    world_points = []
    gaze_images = []
    for i in range(len(points)):
        semantic_waypoint = streamers[i].deproject_pixel(points[i], depth_frames[i])
        waypoint_h = np.append(semantic_waypoint, 1).reshape(4, 1)
        transformed_waypoint = TCR[i] @ waypoint_h
        world_points.append(transformed_waypoint.flatten())
        gaze_images.append(cv2.circle(realsense_imgs[i], (int(points[i][0]), int(points[i][1])), 3, base_link_color[i % len(base_link_color)], -1))

    distance = np.linalg.norm(np.array(world_points[0]) - np.array(world_points[1]))
    if distance > threshold:
        print(f"Potential occlusion detected between points {world_points[0]} and {world_points[1]}. Distance: {distance:.2f} m")
        return True, gaze_images, world_points
    else:   
        return False, None, world_points

def filter_points_by_radius(points, normals, center_point, radius):
    """
    Filters points and normals based on distance from a center point.

    Args:
        points: (N, 3) array of 3D points.
        normals: (N, 3) array of corresponding normals.
        center_point: (3,) array representing the center.
        radius: scalar distance threshold.

    Returns:
        filtered_points: (M, 3) array of filtered points.
        filtered_normals: (M, 3) array of filtered normals.
    """
    points = np.asarray(points)
    normals = np.asarray(normals)
    center_point = np.asarray(center_point)

    distances = np.linalg.norm(points - center_point, axis=1)
    mask = distances <= radius
    filtered_points = points[mask]
    filtered_normals = normals[mask]

    # --- Visualization ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    if filtered_normals.size > 0:
        filtered_pcd.normals = o3d.utility.Vector3dVector(filtered_normals)
    filtered_pcd.paint_uniform_color([1, 0, 0])  # Paint filtered points red

    # Create a bounding box (sphere approximation for visualization)
    bounding_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    bounding_sphere.translate(center_point)
    bounding_sphere.paint_uniform_color([0, 0, 1])  # Paint bounding box blue

    # Visualize
    o3d.visualization.draw_geometries([pcd, filtered_pcd, bounding_sphere])

    return filtered_points, filtered_normals

if __name__ == "__main__":

    base_link_color = [[1, 0.6, 0.8],  [1, 1, 0], [1, 0.5, 0], [0.4, 0, 0.8]]
    streamers = [realsense_streamer_81, realsense_streamer_56]
    results = [capture_and_process_rgbd(streamer) for streamer in streamers]
    pcds, realsense_imgs, depth_frames, depth_images = zip(*results)
    pcds, realsense_imgs, depth_frames, depth_images = np.array(pcds), np.array(realsense_imgs), np.array(depth_frames), np.array(depth_images)
    pixel_selector_81 = PixelSelector()
    pixel_selector_56 = PixelSelector()
    pixels_81, img_81 = pixel_selector_81.run(realsense_imgs[0])
    pixels_56, img_56 = pixel_selector_56.run(realsense_imgs[1])
    pixels = pixels_81
    pixels.extend(pixels_56)
    print(pixels)
    boolean, gaze_images, world_points = select_viewpts(pixels, streamers, depth_frames, realsense_imgs, TCR)
    if gaze_images:
        for i in range(len(gaze_images)):
            cv2.imshow(f"gaze_image_{i}", gaze_images[i])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    '''pcd = pcds[0]
    points_3d_homog = np.vstack((np.array(pcd.points).T, np.ones((1, np.array(pcd.points).shape[0]))))  # Shape: (4, N)
    transformed_points_homog = TCR[0] @ points_3d_homog
    transformed_points = transformed_points_homog[:3, :].T # Extract x, y, z and transpose

    # Convert the NumPy array to an Open3D Vector3dVector
    pcd.points = o3d.utility.Vector3dVector(transformed_points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    print(f"Original points: {np.asarray(pcd.points).shape}, normals: {np.asarray(pcd.normals).shape}")
    points, normals = filter_points_by_radius(np.asarray(pcd.points), np.asarray(pcd.normals), world_points[0], 0.1)
    print(f"Filtered points: {points.shape}, normals: {normals.shape}")
    labels = cluster_normals_with_position(points, normals, n_clusters=5, direction_weight=0.4)
    centroids = get_cluster_centroids(points, normals, labels)
    print(labels)
    print(centroids)

    for i, (pt, n) in enumerate(centroids):
        T = normal_to_pose_matrix(pt, n, pitch_degrees=30)  # You can change pitch here
        centroids[i].append(T)
        # Visualize the cluster
        vis = o3d.visualization.Visualizer()
        window_width = 4988
        window_height = 2742
        vis.create_window(width=window_width, height=window_height)
        visualize_gripper_with_cylinders(vis, T, pcd, base_color=[base_link_color[i % len(base_link_color)]])
        print(f"Cluster {i} Pose Matrix:\n{T}\n, Point {pt}, Normal {n}")
        vis.run()
        vis.destroy_window()
'''
    # Visualize
    #o3d.visualization.draw_geometries([pcd], point_show_normal=True)


