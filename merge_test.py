import open3d as o3d
import numpy as np
import copy

# Function to visualize registration results
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])  # Color source point cloud (orange)
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # Color target point cloud (blue)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


# Function for pairwise registration using ICP
def pairwise_registration(source, target, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


# Function for global registration using RANSAC with color information
def global_registration(source, target, voxel_size):
    # Downsample point clouds
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)

    # Estimate normals
    source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Compute FPFH features
    radius_feature = voxel_size * 5
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    # Perform global registration using RANSAC with color information
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,  # RANSAC n points
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=1000000, confidence=0.9999))
    return result.transformation


def main():
    # Camera serial numbers
    serial_no = '317422075456'  # Camera further from robot arm
    serial_no1 = '317422074281'

    # 1. Load point clouds
    #pcd1 = o3d.io.read_point_cloud(f"./calib/{serial_no}.pcd")
    #pcd2 = o3d.io.read_point_cloud(f"./calib/{serial_no1}.pcd")
    pcd1 = o3d.io.read_point_cloud(f'./calib/segmented{serial_no}.pcd')
    pcd2 = o3d.io.read_point_cloud(f'./calib/segmented{serial_no1}.pcd')
    

    # Visualize the original point clouds
    print("Visualizing pcd1:")
    o3d.visualization.draw_geometries([pcd1])
    print("Visualizing pcd2:")
    o3d.visualization.draw_geometries([pcd2])

    # 2. Load transformation matrices
    #transforms = np.load(f'calib/transforms_{serial_no}.npy', allow_pickle=True).item()
    transforms = np.load(f'calib/transforms.npy', allow_pickle=True).item()
    TCR = transforms[serial_no]['tcr']
    TCR_square = np.eye(4)
    TCR_square[:3, :3] = TCR[:3, :3]
    TCR_square[:3, 3] = TCR[:3, 3]
    print("TCR_square (Camera 1 to Base):\n", TCR_square)

    #transforms1 = np.load(f'calib/transforms_{serial_no1}.npy', allow_pickle=True).item()
    TCR1 = transforms[serial_no1]['tcr']
    TCR_square1 = np.eye(4)
    TCR_square1[:3, :3] = TCR1[:3, :3]
    TCR_square1[:3, 3] = TCR1[:3, 3]
    print("TCR_square1 (Camera 2 to Base):\n", TCR_square1)

    # 3. Calculate relative transformation
    T_camera1_base_inverse = np.linalg.inv(TCR_square)
    T_camera2_camera1 = TCR_square1 @ T_camera1_base_inverse
    print("Relative Transformation (Camera 2 to Camera 1):\n", T_camera2_camera1)

    # 4. Transform point cloud 2
    pcd2_transformed = pcd2.transform(T_camera2_camera1)

    # Visualize the transformed point cloud
    print("Visualizing transformed pcd2 and pcd1:")
    draw_registration_result(pcd2_transformed, pcd1, np.identity(4))

    # 5. Estimate normals for both point clouds
    print("Estimating normals...")
    pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd2_transformed.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # 6. Perform global registration
    voxel_size = 0.05
    print("Performing global registration...")
    transformation_global = global_registration(pcd2_transformed, pcd1, voxel_size)
    print("Global Registration Transformation:\n", transformation_global)

    # Apply the global transformation
    pcd2_global = pcd2_transformed.transform(transformation_global)

    # Visualize the globally aligned point cloud
    print("Visualizing globally aligned pcd2 and pcd1:")
    draw_registration_result(pcd2_global, pcd1, np.identity(4))

    # 7. Refine the transformation using ICP
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5

    # Downsample point clouds
    pcd1_down = pcd1.voxel_down_sample(voxel_size)
    pcd2_global_down = pcd2_global.voxel_down_sample(voxel_size)

    # Perform ICP refinement
    transformation_icp, information_icp = pairwise_registration(
        pcd2_global_down, pcd1_down, max_correspondence_distance_coarse, max_correspondence_distance_fine)
    print("ICP Refinement Transformation:\n", transformation_icp)

    # Apply the refined transformation
    pcd2_refined = pcd2_global.transform(transformation_icp)

    # Visualize the refined point cloud
    print("Visualizing refined pcd2 and pcd1:")
    draw_registration_result(pcd2_refined, pcd1, np.identity(4))

    # 8. Merge point clouds
    pcd_combined = pcd1 + pcd2_refined

    # Visualize the merged point cloud
    print("Visualizing merged point cloud:")
    o3d.visualization.draw_geometries([pcd_combined])

    # Save the merged point cloud (optional)
    o3d.io.write_point_cloud("merged_point_cloud.pcd", pcd_combined)

    return 0


if __name__ == "__main__":
    main()