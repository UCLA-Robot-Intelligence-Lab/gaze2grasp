import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy.spatial.transform import Rotation 

"""
is_position_in_range() is defined in meters in contrast to the is_position_in_range() in multicam which limits the robot motion in mm. 
The generated grasps are also displaced vertically by 157mm to match the position of the finger.
"""

def is_position_in_range(position, x_range=(0.060, 0.550), y_range=(-0.550, 0.550), z_range=(0.020-0.157, 0.600-0.157)):#z_range=(0.020, 0.600)):
    """Check if position is within valid workspace ranges in meters in the pcd ."""
    x, y, z = position
    return (x_range[0] <= x <= x_range[1] and
            y_range[0] <= y <= y_range[1] and
            z_range[0] <= z <= z_range[1])
            
def find_closest_grasp(pred_grasps_cam, pred_gripper_openings, semantic_waypoint, filter_in_range=True):
    """
    Finds the closest grasp to a semantic waypoint, optionally filtering by workspace range.

    Args:
        pred_grasps_cam: Grasp predictions from the camera perspective. Can be a dictionary or a NumPy array.
        pred_gripper_openings: Corresponding gripper opening widths. Must match structure of pred_grasps_cam.
        semantic_waypoint: 3D coordinates (x, y, z) of the semantic waypoint.
        filter_in_range: Boolean to filter out grasps outside the workspace range.

    Returns:
        tuple: (closest_grasp, gripper_opening) where:
            - closest_grasp: 4x4 matrix representing the closest grasp pose
            - gripper_opening: Corresponding gripper opening width
    """

    # Initialize variables to track original indices and openings
    original_indices = []
    grasp_list = []
    opening_list = []

    if isinstance(pred_grasps_cam, dict):
        # Handle dictionary input (multiple grasp sets)
        current_idx = 0
        for k in pred_grasps_cam:
            if pred_grasps_cam[k].size > 0:
                num_grasps = len(pred_grasps_cam[k])
                original_indices.extend(range(current_idx, current_idx + num_grasps))
                grasp_list.append(pred_grasps_cam[k])
                
                # Handle both dictionary and array openings
                if isinstance(pred_gripper_openings, dict):
                    opening_list.append(pred_gripper_openings[k])
                else:
                    opening_list.append(pred_gripper_openings[current_idx:current_idx+num_grasps])
                
                current_idx += num_grasps
        
        if not grasp_list:
            print("No grasps predicted.")
            return None, None
            
        all_grasps = np.concatenate(grasp_list, axis=0)
        all_openings = np.concatenate(opening_list, axis=0)
    else:
        # Handle array input
        all_grasps = pred_grasps_cam
        all_openings = pred_gripper_openings
        original_indices = range(len(all_grasps))

    # Extract grasp centers
    if all_grasps.ndim == 3 and all_grasps.shape[1:] == (4, 4):
        grasp_centers = all_grasps[:, :3, 3]
    elif all_grasps.ndim == 2 and all_grasps.shape[1] >= 3:
        grasp_centers = all_grasps[:, :3]
    elif all_grasps.ndim == 1 and all_grasps.shape[0] >= 3:
        grasp_centers = all_grasps[:3]
        all_grasps = all_grasps.reshape(1, -1)
        all_openings = np.array([all_openings]) if not isinstance(all_openings, np.ndarray) else all_openings.reshape(1)
        original_indices = [0]
    else:
        raise ValueError(f"Unexpected shape for all_grasps: {all_grasps.shape}")

    # Optionally filter grasps by workspace range
    if filter_in_range:
        valid_indices = [i for i, center in enumerate(grasp_centers) if is_position_in_range(center)]
        if not valid_indices:
            print("No grasps within workspace range.")
            return None, None
        grasp_centers = grasp_centers[valid_indices]
        all_grasps = all_grasps[valid_indices]
        all_openings = all_openings[valid_indices]
        original_indices = [original_indices[i] for i in valid_indices]

    # Find nearest grasp
    nbrs = NearestNeighbors(n_neighbors=1).fit(grasp_centers)
    dists, idxs = nbrs.kneighbors(np.array(semantic_waypoint).reshape(1, -1), return_distance=True)
    best_idx = idxs[0][0]

    # Get the closest grasp and opening
    if all_grasps.ndim == 3 and all_grasps.shape[1:] == (4, 4):
        closest_grasp = all_grasps[best_idx]
    elif all_grasps.ndim == 2 and all_grasps.shape[1] >= 3:
        closest_grasp_center = all_grasps[best_idx, :3]
        closest_grasp = np.eye(4)
        closest_grasp[:3, 3] = closest_grasp_center
    else:
        closest_grasp_center = all_grasps[best_idx, :3]
        closest_grasp = np.eye(4)
        closest_grasp[:3, 3] = closest_grasp_center

    gripper_opening = all_openings[best_idx] if len(all_openings) > best_idx else None

    print(f"CLOSEST GRASP (index {original_indices[best_idx]}):", closest_grasp)
    print(f"GRIPPER OPENING: {gripper_opening}")
    return closest_grasp, gripper_opening
def find_distinct_grasps(
    pred_grasps_cam,
    pred_gripper_openings,
    semantic_waypoint,
    n_grasps=3,
    max_distance=0.25,
    filter_in_range=True,
):
    """
    Finds distinct grasps near a gaze point using improved clustering on position and orientation.

    Args:
        pred_grasps_cam: Grasp predictions (dict or array of 4x4 matrices).
        pred_gripper_openings: Gripper widths for each grasp (dict or array).
        semantic_waypoint: 3D gaze point (x, y, z).
        n_grasps: Number of distinct grasps to return.
        max_distance: Max Euclidean distance (meters) from gaze point.
        filter_in_range: If True, filters out-of-range grasps.
        min_orientation_diff: Minimum angular difference (radians) between selected grasps.

    Returns:
        tuple: (distinct_grasps, distinct_openings)
    """
    # --- Data Preparation ---
    if isinstance(pred_grasps_cam, dict):
        # Handle dictionary input (multiple grasp sets)
        all_grasps = np.concatenate(list(pred_grasps_cam.values()), axis=0)
        if isinstance(pred_gripper_openings, dict):
            all_openings = np.concatenate(list(pred_gripper_openings.values()), axis=0)
        else:
            all_openings = pred_gripper_openings  # Assume it's already a flat array
        original_indices = np.arange(len(all_grasps))
    else:
        # Handle array input
        all_grasps = pred_grasps_cam
        all_openings = pred_gripper_openings
        original_indices = np.arange(len(all_grasps))

    # --- Extract positions and orientations ---
    grasp_positions = all_grasps[:, :3, 3]
    grasp_rotations = Rotation.from_matrix(all_grasps[:, :3, :3])
    grasp_quats = grasp_rotations.as_quat()  # [x, y, z, w] format

    # --- Filtering ---
    if filter_in_range:
        valid_mask = np.array([is_position_in_range(p) for p in grasp_positions])
        grasp_positions = grasp_positions[valid_mask]
        grasp_quats = grasp_quats[valid_mask]
        all_grasps = all_grasps[valid_mask]
        all_openings = all_openings[valid_mask]
        original_indices = original_indices[valid_mask]

    # --- Find grasps near the semantic waypoint ---
    if len(grasp_positions) == 0:
        print("No valid grasp positions after filtering.")
        return [], []

    n_neighbors_val = min(50, len(grasp_positions))
    nbrs = NearestNeighbors(n_neighbors=max(1, n_neighbors_val)).fit(grasp_positions)
    dists, idxs = nbrs.kneighbors([semantic_waypoint], return_distance=True)
    nearby_mask = dists[0] <= max_distance
    nearby_indices = idxs[0][nearby_mask]

    if len(nearby_indices) == 0:
        print("No grasps found within max_distance.")
        return [], []

    nearby_grasps = all_grasps[nearby_indices]
    nearby_positions = grasp_positions[nearby_indices]
    nearby_quats = grasp_quats[nearby_indices]
    nearby_openings = all_openings[nearby_indices]

    # --- Feature Engineering ---
    position_weight = 0.2
    pos_features = (nearby_positions - semantic_waypoint) / max_distance * position_weight  # Normalized
    quat_features = nearby_quats * (1.0 - position_weight)  # Quaternions are already unit vectors
    features = np.hstack([pos_features, quat_features])

    # --- Clustering ---
    n_clusters = min(n_grasps, len(nearby_grasps))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(features) # Added n_init for stability
    cluster_labels = kmeans.labels_

    # --- Select Distinct Grasps ---
    distinct_grasps = []
    distinct_openings = []

    for cluster_id in range(n_clusters):
        cluster_mask = (cluster_labels == cluster_id)
        cluster_grasps = nearby_grasps[cluster_mask]
        cluster_openings = nearby_openings[cluster_mask]

        if len(cluster_grasps) > 0:
            # Select the grasp with median opening width in the cluster
            median_idx = np.argsort(cluster_openings)[len(cluster_openings) // 2]
            selected_grasp = cluster_grasps[median_idx]
            selected_opening = cluster_openings[median_idx]

            distinct_grasps.append(selected_grasp)
            distinct_openings.append(selected_opening)

    print(f"Selected {len(distinct_grasps)} distinct grasps.")
    return distinct_grasps, distinct_openings