import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

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

def find_distinct_grasps(pred_grasps_cam, pred_gripper_openings, semantic_waypoint, n_grasps=3, max_distance=0.25, filter_in_range=True):
    """
    Finds distinct grasps near a gaze point using clustering on both position and orientation,
    within a maximum Euclidean distance, optionally filtering by workspace range.

    Args:
        pred_grasps_cam: Grasp predictions from the camera perspective. Can be a dictionary or a NumPy array.
        pred_gripper_openings: Corresponding gripper opening widths for each grasp.
        semantic_waypoint: 3D coordinates (x, y, z) of the semantic waypoint.
        n_grasps: Number of distinct grasps to return.
        max_distance: Maximum Euclidean distance (in meters) from the gaze point for grasps to be considered.
        filter_in_range: Boolean to filter out grasps outside the workspace range.

    Returns:
        tuple: (distinct_grasps, distinct_openings, original_indices) where:
            - distinct_grasps: List of 4x4 matrices representing the distinct grasp poses
            - distinct_openings: List of corresponding gripper opening widths
            - original_indices: List of original indices in the input predictions
    """

    # Combine all grasps into a single array while preserving original indices
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
                opening_list.append(pred_gripper_openings[k] if isinstance(pred_gripper_openings, dict) else pred_gripper_openings[current_idx:current_idx+num_grasps])
                current_idx += num_grasps
        if not grasp_list:
            print("No grasps predicted.")
            return None, None, None
        all_grasps = np.concatenate(grasp_list, axis=0)
        all_openings = np.concatenate(opening_list, axis=0)
    else:
        # Handle array input
        all_grasps = pred_grasps_cam
        all_openings = pred_gripper_openings
        original_indices = range(len(all_grasps))

    # Extract grasp centers and orientations
    if all_grasps.ndim == 3 and all_grasps.shape[1:] == (4, 4):
        grasp_centers = all_grasps[:, :3, 3]
        # Extract rotation matrix components for clustering
        grasp_orientations = all_grasps[:, :3, :3].reshape(len(all_grasps), -1)  # Flatten rotation matrices
    elif all_grasps.ndim == 2 and all_grasps.shape[1] >= 3:
        grasp_centers = all_grasps[:, :3]
        grasp_orientations = np.zeros((len(all_grasps), 9))  # Dummy orientations if not available
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
        grasp_orientations = grasp_orientations[valid_indices]

    # Combine position and orientation features for clustering (with scaling)
    position_scale = 1.0  # meters
    orientation_scale = 0.5  # smaller weight for orientation
    features = np.hstack([
        grasp_centers * position_scale,
        grasp_orientations * orientation_scale
    ])

    # Find grasps near the gaze point within max_distance
    nbrs = NearestNeighbors(n_neighbors=min(50, len(grasp_centers))).fit(grasp_centers)
    dists, idxs = nbrs.kneighbors(np.array(semantic_waypoint).reshape(1, -1), return_distance=True)
    
    # Filter grasps within max_distance
    nearby_mask = dists[0] <= max_distance
    nearby_indices = idxs[0][nearby_mask]
    nearby_grasps = all_grasps[nearby_indices]
    nearby_openings = all_openings[nearby_indices]
    nearby_features = features[nearby_indices]

    if len(nearby_grasps) == 0:
        print("No grasps found within the specified distance.")
        return None, None, None

    # Cluster the nearby grasps using both position and orientation
    kmeans = KMeans(n_clusters=min(n_grasps, len(nearby_grasps)), random_state=0).fit(nearby_features)
    
    # Find the most central grasp in each cluster
    distinct_grasps = []
    distinct_openings = []
    
    for cluster_id in range(kmeans.n_clusters):
        cluster_mask = (kmeans.labels_ == cluster_id)
        cluster_grasps = nearby_grasps[cluster_mask]
        cluster_features = nearby_features[cluster_mask]
        
        # Find the grasp closest to the cluster center
        cluster_center = kmeans.cluster_centers_[cluster_id]
        nbrs = NearestNeighbors(n_neighbors=1).fit(cluster_features)
        _, idx = nbrs.kneighbors(cluster_center.reshape(1, -1), return_distance=True)
        
        # Get the selected grasp and its metadata
        selected_idx = idx[0][0]
        distinct_grasps.append(cluster_grasps[selected_idx])
        distinct_openings.append(nearby_openings[cluster_mask][selected_idx])

    print(f"Found {len(distinct_grasps)} distinct grasps with openings: {distinct_openings}")
    return distinct_grasps, distinct_openings
