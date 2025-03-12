import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

def find_closest_grasp(pred_grasps_cam, gaze, depth_frame, realsense_streamer):
    """
    Finds the closest grasp to a gaze point in a depth frame.

    Args:
        pred_grasps_cam: Grasp predictions from the camera perspective. Can be a dictionary or a NumPy array.
        gaze: Gaze coordinates (x, y) in pixels.
        depth_frame: Depth frame data.
        realsense_streamer: Instance of the RealsenseStreamer class for deprojection.

    Returns:
        closest_grasp: 4x4 matrix representing the closest grasp pose.
    """
    try:
        semantic_waypoint = realsense_streamer.deproject(gaze, depth_frame)
    except:
        semantic_waypoint = realsense_streamer.deproject_pixel(gaze, depth_frame)

    if isinstance(pred_grasps_cam, dict):
        all_grasps = []
        for k in pred_grasps_cam:
            if pred_grasps_cam[k].size > 0:
                all_grasps.append(pred_grasps_cam[k])
        if not all_grasps:
            print("No grasps predicted.")
            return None  # Return None if no grasps are found
        all_grasps = np.concatenate(all_grasps, axis=0)
    else:
        all_grasps = pred_grasps_cam

    # Extract grasp centers
    if all_grasps.ndim == 3 and all_grasps.shape[1:] == (4, 4):
        grasp_centers = all_grasps[:, :3, 3]
    elif all_grasps.ndim == 2 and all_grasps.shape[1] >= 3:
        grasp_centers = all_grasps[:, :3]
    elif all_grasps.ndim == 1 and all_grasps.shape[0] >= 3:
        grasp_centers = all_grasps[:3]
        all_grasps = all_grasps.reshape(1, -1)
    else:
        raise ValueError(f"Unexpected shape for all_grasps: {all_grasps.shape}")

    nbrs = NearestNeighbors(n_neighbors=1).fit(grasp_centers)
    dists, idxs = nbrs.kneighbors(np.array(semantic_waypoint).reshape(1, -1), return_distance=True)
    best_idx = idxs[0][0]

    # Access the correct grasp
    if all_grasps.ndim == 3 and all_grasps.shape[1:] == (4, 4):
        closest_grasp = all_grasps[best_idx]
    elif all_grasps.ndim == 2 and all_grasps.shape[1] >= 3:
        closest_grasp_center = all_grasps[best_idx, :3]
        closest_grasp = np.eye(4)
        closest_grasp[:3, 3] = closest_grasp_center
    elif all_grasps.ndim == 1 and all_grasps.shape[0] >= 3:
        closest_grasp_center = all_grasps[best_idx, :3]
        closest_grasp = np.eye(4)
        closest_grasp[:3, 3] = closest_grasp_center
    else:
        raise ValueError("Unexpected shape after nearest neighbor search.")

    print("CLOSEST GRASP:", closest_grasp)
    return closest_grasp

def find_distinct_grasps(pred_grasps_cam, gaze, depth_frame, realsense_streamer, n_grasps=3, max_distance=0.25):
    """
    Finds distinct grasps near a gaze point using clustering, within a maximum Euclidean distance.

    Args:
        pred_grasps_cam: Grasp predictions from the camera perspective. Can be a dictionary or a NumPy array.
        gaze: Gaze coordinates (x, y) in pixels.
        depth_frame: Depth frame data.
        realsense_streamer: Instance of the RealsenseStreamer class for deprojection.
        n_grasps: Number of distinct grasps to return.
        max_distance: Maximum Euclidean distance (in meters) from the gaze point for grasps to be considered.

    Returns:
        distinct_grasps: List of 4x4 matrices representing the distinct grasp poses.
    """
    try:
        semantic_waypoint = realsense_streamer.deproject(gaze, depth_frame)
    except:
        semantic_waypoint = realsense_streamer.deproject_pixel(gaze, depth_frame)

    # Combine all grasps into a single array
    if isinstance(pred_grasps_cam, dict):
        all_grasps = []
        for k in pred_grasps_cam:
            if pred_grasps_cam[k].size > 0:
                all_grasps.append(pred_grasps_cam[k])
        if not all_grasps:
            print("No grasps predicted.")
            return None  # Return None if no grasps are found
        all_grasps = np.concatenate(all_grasps, axis=0)
    else:
        all_grasps = pred_grasps_cam

    # Extract grasp centers
    if all_grasps.ndim == 3 and all_grasps.shape[1:] == (4, 4):
        grasp_centers = all_grasps[:, :3, 3]
    elif all_grasps.ndim == 2 and all_grasps.shape[1] >= 3:
        grasp_centers = all_grasps[:, :3]
    elif all_grasps.ndim == 1 and all_grasps.shape[0] >= 3:
        grasp_centers = all_grasps[:3]
        all_grasps = all_grasps.reshape(1, -1)
    else:
        raise ValueError(f"Unexpected shape for all_grasps: {all_grasps.shape}")

    # Find grasps near the gaze point within max_distance
    nbrs = NearestNeighbors(n_neighbors=min(50, len(grasp_centers))).fit(grasp_centers)
    dists, idxs = nbrs.kneighbors(np.array(semantic_waypoint).reshape(1, -1), return_distance=True)
    
    # Filter grasps within max_distance
    nearby_mask = dists[0] <= max_distance
    nearby_grasps = all_grasps[idxs[0][nearby_mask]]

    if len(nearby_grasps) == 0:
        print("No grasps found within the specified distance.")
        return None

    # Cluster the nearby grasps
    kmeans = KMeans(n_clusters=min(n_grasps, len(nearby_grasps)), random_state=0).fit(nearby_grasps[:, :3, 3])
    cluster_centers = kmeans.cluster_centers_

    # Find the closest grasp in each cluster
    distinct_grasps = []
    for cluster_center in cluster_centers:
        nbrs = NearestNeighbors(n_neighbors=1).fit(nearby_grasps[:, :3, 3])
        _, idx = nbrs.kneighbors(cluster_center.reshape(1, -1), return_distance=True)
        distinct_grasps.append(nearby_grasps[idx[0][0]])

    print("DISTINCT GRASPS:", distinct_grasps)
    return distinct_grasps
