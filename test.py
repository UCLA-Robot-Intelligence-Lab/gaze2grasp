from homography import HomographyManager # Import HomographyManager
import sys
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import time
import os

homography_manager = HomographyManager()
try:
    homography_manager.start_camera()
except Exception as e:
    print(f"Error starting RealSense camera: {e}")
    # Handle the error appropriately, e.g., continue without homography if needed
    sys.exit("RealSense camera failed to start, exiting.") # Exit if RealSense camera fails to start

### IF YOU WANT TO PARSE THE 3D POINT CLOUD ###
homography_manager.get_point_cloud(gaze_coordinates = [330,330])

# Load the point cloud data from the .npz file
data = np.load('3dpoint_data.npz')
points = data['xyz']

# Create an Open3D point cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# (Optional) Estimate normals (for better visualization)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
pcd.orient_normals_towards_camera_location(camera_location=[0., 0., 0.])

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.1,  # Adjust size as needed
    origin=[0, 0, 0] #origin
)

# Visualize the point cloud and coordinate frame together
o3d.visualization.draw_geometries([pcd, mesh_frame])

time.sleep(10)
'''


### IF YOU WANT TO PARSE THE 2D DEPTH WITH SEGMENTATION MAP USING FASTSAM ###

homography_manager.fetch_3D_scene(point_prompt = "[[330,330]]")

# Load the saved data
data = np.load("depth_data.npz")
depth_image = data["depth"]
K = data["K"]
segmap = data["segmap"]

plt.figure()
plt.imshow(segmap)
plt.title("Seg Map")
plt.show(block=False)  # Keep the window open
# Show depth map in a separate window
plt.figure()
plt.imshow(depth_image, cmap="gray")
plt.colorbar(label="Depth (m)")
plt.title("Depth Map")
plt.show(block=False)  # Keep the window open
time.sleep(10)
# Convert depth map to point cloud
h, w = depth_image.shape
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

points = []
for v in range(h):
    for u in range(w):
        z = depth_image[v, u]
        if z > 0:  # Ignore invalid depth values
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append([x, y, z])

# Convert to Open3D format
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(points))

# Open point cloud visualization in a separate window
o3d.visualization.draw_geometries([pcd])

'''
