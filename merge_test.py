import open3d as o3d
import numpy as np

# Camera serial numbers
serial_no = '317422075456'  # Camera further from robot arm
serial_no1 = '317422074281'

# 1. Load point clouds
pcd1 = o3d.io.read_point_cloud(f"./calib/{serial_no}.pcd")
pcd2 = o3d.io.read_point_cloud(f"./calib/{serial_no1}.pcd")

# Visualize the original point clouds
print("Visualizing pcd1:")
o3d.visualization.draw_geometries([pcd1])
print("Visualizing pcd2:")
o3d.visualization.draw_geometries([pcd2])

# 2. Load transformation matrices
transforms = np.load(f'calib/transforms_{serial_no}.npy', allow_pickle=True).item()
TCR = transforms[serial_no]['tcr']
TCR_square = np.eye(4)
TCR_square[:3, :3] = TCR[:3, :3]
TCR_square[:3, 3] = TCR[:3, 3]
print("TCR_square (Camera 1 to Base):\n", TCR_square)

transforms1 = np.load(f'calib/transforms_{serial_no1}.npy', allow_pickle=True).item()
TCR1 = transforms1[serial_no1]['tcr']
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
o3d.visualization.draw_geometries([pcd1, pcd2_transformed])

# 5. Refine the transformation using ICP
print("Running ICP to refine the transformation...")
threshold = 0.02  # Maximum correspondence distance
reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd2_transformed, pcd1, threshold, T_camera2_camera1,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
)
print("ICP transformation refinement:\n", reg_p2p.transformation)

# Apply the refined transformation
pcd2_refined = pcd2_transformed.transform(reg_p2p.transformation)

# Visualize the refined point cloud
print("Visualizing refined pcd2 and pcd1:")
o3d.visualization.draw_geometries([pcd1, pcd2_refined])

# 6. Merge point clouds
merged_pcd = pcd1 + pcd2_refined

# 7. Visualize the merged point cloud
print("Visualizing merged point cloud:")
o3d.visualization.draw_geometries([merged_pcd])

# 8. Save the merged point cloud (optional)
o3d.io.write_point_cloud("merged_point_cloud.pcd", merged_pcd)