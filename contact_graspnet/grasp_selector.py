import numpy as np
from sklearn.neighbors import NearestNeighbors

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