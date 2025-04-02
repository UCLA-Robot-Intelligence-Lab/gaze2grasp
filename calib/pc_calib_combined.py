import open3d as o3d
import numpy as np

# Load your point cloud
pcd = o3d.io.read_point_cloud("/home/u-ril/gaze2grasp/combined_pcd.pcd") #replace with your pcd

# Interactive visualization
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
vis.run()  # This will open the interactive window
vis.destroy_window()

# Get the ViewControl after manual adjustment
view_control = vis.get_view_control()

# Get the pinhole camera parameters
pinhole_camera_parameters = view_control.convert_to_pinhole_camera_parameters()

# Print the parameters (or save them to a file)
print("Camera Parameters:")
print("  Intrinsic:")
print(pinhole_camera_parameters.intrinsic)
print("  Extrinsic:")
print(pinhole_camera_parameters.extrinsic)

extrinsic = pinhole_camera_parameters.extrinsic

# Save the extrinsic matrix as a .npy file
np.save("./calib/extrinsic_combined2.npy", pinhole_camera_parameters.extrinsic)

print("Extrinsic matrix saved to extrinsic_combined2.npy")
