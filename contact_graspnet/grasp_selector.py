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

def translation_distance(g1, g2):
    """Calculates the Euclidean distance between the translation vectors of two grasps."""
    return np.linalg.norm(g1[:3, 3] - g2[:3, 3])

def rotation_distance_quaternion(g1, g2):
    """Calculates the Euclidean distance between the quaternion representations of two rotations."""
    q1 = Rotation.from_matrix(g1[:3, :3]).as_quat()
    q2 = Rotation.from_matrix(g2[:3, :3]).as_quat()
    return np.linalg.norm(q1 - q2)

def find_median_grasp_with_opening(grasps_with_openings, weight_t=1.0, weight_r=1.0):
    """
    Finds the median grasp (and its opening) from a list of grasp matrices and their openings
    based on a weighted distance metric. The median grasp is the one that minimizes the
    sum of its distances to all other grasps.

    Args:
        grasps_with_openings (list of tuples): A list where each tuple contains:
            - grasp (np.ndarray): A 4x4 grasp matrix.
            - opening (float): The corresponding gripper opening width.
        weight_t (float): Weight for the translation distance.
        weight_r (float): Weight for the rotation distance.

    Returns:
        tuple or None: A tuple containing the median grasp matrix and its opening,
                       or None if the input list is empty.
    """
    if not grasps_with_openings:
        return None

    num_grasps = len(grasps_with_openings)
    total_distances = np.zeros(num_grasps)

    for i in range(num_grasps):
        grasp_i, _ = grasps_with_openings[i]
        for j in range(num_grasps):
            grasp_j, _ = grasps_with_openings[j]
            dist_t = translation_distance(grasp_i, grasp_j)
            dist_r = rotation_distance_quaternion(grasp_i, grasp_j)
            total_distances[i] += weight_t * dist_t + weight_r * dist_r

    median_index = np.argmin(total_distances)
    median_grasp, median_opening = grasps_with_openings[median_index]
    return median_grasp, median_opening

def find_median_grasp(pred_grasps_cam, pred_gripper_openings):
    """
    Finds the median grasp (and its opening) from the input grasp predictions.

    Args:
        pred_grasps_cam: Grasp predictions from the camera perspective. Can be a dictionary or a NumPy array.
        pred_gripper_openings: Corresponding gripper opening widths. Must match structure of pred_grasps_cam.

    Returns:
        tuple or None: A tuple containing the median grasp matrix and its opening,
                       or None if no valid grasps are found.
    """
    grasp_opening_pairs = []

    if isinstance(pred_grasps_cam, dict):
        for k in pred_grasps_cam:
            if pred_grasps_cam[k].size > 0:
                grasps = pred_grasps_cam[k]
                openings = pred_gripper_openings[k] if isinstance(pred_gripper_openings, dict) else pred_gripper_openings
                if grasps.ndim == 3:
                    for i in range(len(grasps)):
                        grasp_opening_pairs.append((grasps[i], openings[i]))
                elif grasps.ndim == 2 and grasps.shape[1] >= 3:
                    for i in range(len(grasps)):
                        grasp_matrix = np.eye(4)
                        grasp_matrix[:3, 3] = grasps[i, :3]
                        grasp_opening_pairs.append((grasp_matrix, openings[i]))
                elif grasps.ndim == 1 and grasps.shape[0] >= 3:
                    grasp_matrix = np.eye(4)
                    grasp_matrix[:3, 3] = grasps[:3]
                    grasp_opening_pairs.append((grasp_matrix, openings))
                else:
                    raise ValueError(f"Unexpected shape for pred_grasps_cam[{k}]: {grasps.shape}")
    else:
        grasps = pred_grasps_cam
        openings = pred_gripper_openings
        if grasps.ndim == 3:
            for i in range(len(grasps)):
                grasp_opening_pairs.append((grasps[i], openings[i]))
        elif grasps.ndim == 2 and grasps.shape[1] >= 3:
            for i in range(len(grasps)):
                grasp_matrix = np.eye(4)
                grasp_matrix[:3, 3] = grasps[i, :3]
                grasp_opening_pairs.append((grasp_matrix, openings[i]))
        elif grasps.ndim == 1 and grasps.shape[0] >= 3:
            grasp_matrix = np.eye(4)
            grasp_matrix[:3, 3] = grasps[:3]
            grasp_opening_pairs.append((grasp_matrix, openings))
        else:
            raise ValueError(f"Unexpected shape for pred_grasps_cam: {grasps.shape}")

    if not grasp_opening_pairs:
        print("No valid grasps found to calculate median.")
        return None

    median_grasp, median_opening = find_median_grasp_with_opening(grasp_opening_pairs)
    print("MEDIAN GRASP:", median_grasp)
    print("MEDIAN GRIPPER OPENING:", median_opening)
    return median_grasp, median_opening
def filter_grasps_by_grasp_centre_distance(pred_grasps_cam, pred_gripper_openings, semantic_waypoint, approach_distance=0.9, max_distance=0.1):
    """
    Filters out grasp predictions whose calculated grasp centre (with approach offset)
    is too far from a semantic waypoint and returns the associated gripper openings.

    Args:
        pred_grasps_cam: Grasp predictions from the camera perspective. Can be a dictionary or a NumPy array
                         where each grasp is represented as a 4x4 homogeneous transformation matrix.
        pred_gripper_openings: Corresponding gripper opening widths. Must match the structure of pred_grasps_cam.
        semantic_waypoint: 3D coordinates (x, y, z) of the semantic waypoint.
        approach_distance: The distance along the approach vector to offset from the translation.
        max_distance: Maximum allowed Euclidean distance between the calculated grasp centre
                      and the semantic waypoint.

    Returns:
        tuple: A tuple containing two lists:
            - filtered_grasps: A list of grasp poses (4x4 matrices) whose calculated grasp centre
                               is within the specified distance.
            - filtered_openings: A list of corresponding gripper opening widths for the filtered grasps.
    """

    filtered_grasps = []
    filtered_openings = []

    def get_grasp_centre(grasp):
        """Calculates the grasp centre from a 4x4 grasp pose."""
        R = grasp[:3, :3]
        t = grasp[:3, 3]
        approach_vector_unit = R[:, 2]  # Z-axis is usually the approach direction
        grasp_centre = t + approach_distance * approach_vector_unit
        return grasp_centre

    if isinstance(pred_grasps_cam, dict):
        for k, grasps in pred_grasps_cam.items():
            openings = pred_gripper_openings.get(k)
            if openings is None:
                print(f"Warning: No corresponding gripper openings found for key {k}. Skipping.")
                continue
            for i, grasp in enumerate(grasps):
                if grasp.shape == (4, 4):
                    calculated_grasp_centre = get_grasp_centre(grasp)
                    distance = np.linalg.norm(calculated_grasp_centre - np.array(semantic_waypoint))
                    if distance <= max_distance:
                        filtered_grasps.append(grasp)
                        filtered_openings.append(openings[i])
                else:
                    print(f"Warning: Unexpected grasp shape {grasp.shape} for key {k} at index {i}. Skipping.")
    elif isinstance(pred_grasps_cam, np.ndarray):
        for i, grasp in enumerate(pred_grasps_cam):
            if i >= len(pred_gripper_openings):
                print(f"Warning: Missing gripper opening for grasp at index {i}. Skipping.")
                continue
            if grasp.shape == (4, 4):
                calculated_grasp_centre = get_grasp_centre(grasp)
                distance = np.linalg.norm(calculated_grasp_centre - np.array(semantic_waypoint))
                if distance <= max_distance:
                    filtered_grasps.append(grasp)
                    filtered_openings.append(pred_gripper_openings[i])
            else:
                print(f"Warning: Unexpected grasp shape {grasp.shape} at index {i}. Skipping.")
    else:
        print("Warning: Unexpected type for pred_grasps_cam. Expected dict or NumPy array of 4x4 matrices.")

    return filtered_grasps, filtered_openings

def find_closest_grasp(pred_grasps_cam, pred_gripper_openings, semantic_waypoint, approach_distance=0.9, filter_in_range=True):
    # accounted for the translation distance of the grasp centre to the semantic waypoint
    """
    Finds the closest grasp to a semantic waypoint based on the distance of their
    calculated grasp centres (with approach offset), optionally filtering by workspace range.

    Args:
        pred_grasps_cam: Grasp predictions from the camera perspective. Can be a dictionary or a NumPy array
                         where each grasp is represented as a 4x4 homogeneous transformation matrix.
        pred_gripper_openings: Corresponding gripper opening widths. Must match structure of pred_grasps_cam.
        semantic_waypoint: 3D coordinates (x, y, z) of the semantic waypoint.
        approach_distance: The distance along the approach vector to offset from the translation.
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
    grasp_centres = []

    def get_grasp_centre(grasp):
        """Calculates the grasp centre from a 4x4 grasp pose."""
        R = grasp[:3, :3]
        t = grasp[:3, 3]
        approach_vector_unit = R[:, 2]  # Z-axis is usually the approach direction
        grasp_centre = t + approach_distance * approach_vector_unit
        return grasp_centre

    if isinstance(pred_grasps_cam, dict):
        # Handle dictionary input (multiple grasp sets)
        current_idx = 0
        for k, grasps in pred_grasps_cam.items():
            if grasps.size > 0:
                num_grasps = len(grasps)
                original_indices.extend(range(current_idx, current_idx + num_grasps))
                grasp_list.append(grasps)
                grasp_centres.extend([get_grasp_centre(g) for g in grasps])

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
        all_grasp_centres = np.array(grasp_centres)

    elif isinstance(pred_grasps_cam, np.ndarray):
        # Handle array input
        all_grasps = pred_grasps_cam
        all_openings = pred_gripper_openings
        original_indices = range(len(all_grasps))
        all_grasp_centres = np.array([get_grasp_centre(g) for g in all_grasps])

    else:
        raise ValueError(f"Unexpected type for pred_grasps_cam: {type(pred_grasps_cam)}")

    # Optionally filter grasps by workspace range based on grasp centre
    if filter_in_range:
        valid_indices = [i for i, center in enumerate(all_grasp_centres) if is_position_in_range(center)]
        if not valid_indices:
            print("No grasps within workspace range (based on grasp centre).")
            return None, None
        filtered_grasp_centres = all_grasp_centres[valid_indices]
        filtered_grasps = all_grasps[valid_indices]
        filtered_openings = all_openings[valid_indices]
        filtered_original_indices = [original_indices[i] for i in valid_indices]
    else:
        filtered_grasp_centres = all_grasp_centres
        filtered_grasps = all_grasps
        filtered_openings = all_openings
        filtered_original_indices = original_indices

    # Find nearest grasp based on the filtered grasp centres
    if filtered_grasp_centres.size > 0:
        nbrs = NearestNeighbors(n_neighbors=1).fit(filtered_grasp_centres)
        dists, idxs = nbrs.kneighbors(np.array(semantic_waypoint).reshape(1, -1), return_distance=True)
        best_idx_filtered = idxs[0][0]

        closest_grasp = filtered_grasps[best_idx_filtered]
        gripper_opening = filtered_openings[best_idx_filtered] if len(filtered_openings) > best_idx_filtered else None
        original_index = filtered_original_indices[best_idx_filtered]

        print(f"CLOSEST GRASP (index {original_index}):", closest_grasp)
        print(f"GRIPPER OPENING: {gripper_opening}")
        return closest_grasp, gripper_opening
    else:
        print("No valid grasps found after filtering.")
        return None, None
    
def find_distinct_grasps(
    pred_grasps_cam,
    pred_gripper_openings,
    semantic_waypoint,
    approach_distance=0.9,
    n_grasps=3,
    max_distance=0.25,
    filter_in_range=True,
):
    """
    Finds distinct grasps near a gaze point using improved clustering on grasp centre
    and orientation.

    Args:
        pred_grasps_cam: Grasp predictions (dict or array of 4x4 matrices).
        pred_gripper_openings: Gripper widths for each grasp (dict or array).
        semantic_waypoint: 3D gaze point (x, y, z).
        approach_distance: The distance along the approach vector to offset from the translation.
        n_grasps: Number of distinct grasps to return.
        max_distance: Max Euclidean distance (meters) from gaze point to the grasp centre.
        filter_in_range: If True, filters out-of-range grasps based on grasp centre.

    Returns:
        tuple: (distinct_grasps, distinct_openings)
    """
    # --- Data Preparation ---
    if isinstance(pred_grasps_cam, dict):
        all_grasps = np.concatenate(list(pred_grasps_cam.values()), axis=0)
        if isinstance(pred_gripper_openings, dict):
            all_openings = np.concatenate(list(pred_gripper_openings.values()), axis=0)
        else:
            all_openings = pred_gripper_openings
        original_indices = np.arange(len(all_grasps))
    else:
        all_grasps = pred_grasps_cam
        all_openings = pred_gripper_openings
        original_indices = np.arange(len(all_grasps))

    def get_grasp_centre(grasp):
        """Calculates the grasp centre from a 4x4 grasp pose."""
        R = grasp[:3, :3]
        t = grasp[:3, 3]
        approach_vector_unit = R[:, 2]
        grasp_centre = t + approach_distance * approach_vector_unit
        return grasp_centre

    # --- Calculate Grasp Centres and Orientations ---
    grasp_centres = np.array([get_grasp_centre(g) for g in all_grasps])
    grasp_rotations = Rotation.from_matrix(all_grasps[:, :3, :3])
    grasp_quats = grasp_rotations.as_quat()  # [x, y, z, w] format

    # --- Filtering based on Grasp Centre ---
    if filter_in_range:
        valid_mask = np.array([is_position_in_range(gc) for gc in grasp_centres])
        grasp_centres = grasp_centres[valid_mask]
        grasp_quats = grasp_quats[valid_mask]
        all_grasps = all_grasps[valid_mask]
        all_openings = all_openings[valid_mask]
        original_indices = original_indices[valid_mask]

    # --- Find grasps near the semantic waypoint (based on grasp centre) ---
    if len(grasp_centres) == 0:
        print("No valid grasp centres after filtering.")
        return [], []

    n_neighbors_val = min(50, len(grasp_centres))
    nbrs = NearestNeighbors(n_neighbors=max(1, n_neighbors_val)).fit(grasp_centres)
    dists, idxs = nbrs.kneighbors([semantic_waypoint], return_distance=True)
    nearby_mask = dists[0] <= max_distance
    nearby_indices = idxs[0][nearby_mask]

    if len(nearby_indices) == 0:
        print("No grasps found within max_distance (based on grasp centre).")
        return [], []

    nearby_grasps = all_grasps[nearby_indices]
    nearby_centres = grasp_centres[nearby_indices]
    nearby_quats = grasp_quats[nearby_indices]
    nearby_openings = all_openings[nearby_indices]

    # --- Feature Engineering ---
    position_weight = 0.2
    pos_features = (nearby_centres - semantic_waypoint) / max_distance * position_weight  # Normalized
    quat_features = nearby_quats * (1.0 - position_weight)  # Quaternions are already unit vectors
    features = np.hstack([pos_features, quat_features])

    # --- Clustering ---
    n_clusters = min(n_grasps, len(nearby_grasps))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(features)
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

    print(f"Selected {len(distinct_grasps)} distinct grasps (based on grasp centre).")
    return distinct_grasps, distinct_openings