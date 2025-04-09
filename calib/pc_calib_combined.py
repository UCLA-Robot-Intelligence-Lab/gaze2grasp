import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("/home/u-ril/gaze2grasp/calib/combined_pcd.pcd") 
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
vis.run()  # This will open the interactive window
vis.destroy_window()
view_control = vis.get_view_control()
pinhole_camera_parameters = view_control.convert_to_pinhole_camera_parameters()
extrinsic = pinhole_camera_parameters.extrinsic
intrinsic_data = {
    "width": pinhole_camera_parameters.intrinsic.width,
    "height": pinhole_camera_parameters.intrinsic.height,
    "intrinsic_matrix": np.asarray(pinhole_camera_parameters.intrinsic.intrinsic_matrix)
}
np.save("./calib/extrinsic_combined1.npy", pinhole_camera_parameters.extrinsic)
np.save("./calib/intrinsic1.npy", intrinsic_data)

pcd = o3d.io.read_point_cloud("/home/u-ril/gaze2grasp/calib/combined_pcd.pcd")
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
vis.run()  # This will open the interactive window
vis.destroy_window()
view_control = vis.get_view_control()
pinhole_camera_parameters = view_control.convert_to_pinhole_camera_parameters()
extrinsic = pinhole_camera_parameters.extrinsic
intrinsic_data = {
    "width": pinhole_camera_parameters.intrinsic.width,
    "height": pinhole_camera_parameters.intrinsic.height,
    "intrinsic_matrix": np.asarray(pinhole_camera_parameters.intrinsic.intrinsic_matrix)
}
np.save("./calib/extrinsic_combined2.npy", pinhole_camera_parameters.extrinsic)
np.save("./calib/intrinsic2.npy", intrinsic_data)

pcd = o3d.io.read_point_cloud("/home/u-ril/gaze2grasp/calib/0.pcd")
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
vis.run()  # This will open the interactive window
vis.destroy_window()
view_control = vis.get_view_control()
pinhole_camera_parameters = view_control.convert_to_pinhole_camera_parameters()
extrinsic = pinhole_camera_parameters.extrinsic
intrinsic_data = {
    "width": pinhole_camera_parameters.intrinsic.width,
    "height": pinhole_camera_parameters.intrinsic.height,
    "intrinsic_matrix": np.asarray(pinhole_camera_parameters.intrinsic.intrinsic_matrix)
}
np.save("./calib/extrinsic_combined3.npy", pinhole_camera_parameters.extrinsic)
np.save("./calib/intrinsic3.npy", intrinsic_data)

pcd = o3d.io.read_point_cloud("/home/u-ril/gaze2grasp/calib/1.pcd")
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
vis.run()  # This will open the interactive window
vis.destroy_window()
view_control = vis.get_view_control()
pinhole_camera_parameters = view_control.convert_to_pinhole_camera_parameters()
extrinsic = pinhole_camera_parameters.extrinsic
intrinsic_data = {
    "width": pinhole_camera_parameters.intrinsic.width,
    "height": pinhole_camera_parameters.intrinsic.height,
    "intrinsic_matrix": np.asarray(pinhole_camera_parameters.intrinsic.intrinsic_matrix)
}
np.save("./calib/extrinsic_combined4.npy", pinhole_camera_parameters.extrinsic)
np.save("./calib/intrinsic4.npy", intrinsic_data)



