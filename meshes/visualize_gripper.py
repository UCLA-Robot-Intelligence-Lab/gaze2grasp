import open3d as o3d
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

def create_transformation_matrix(xyz, rpy):
    """
    Create a 4x4 transformation matrix from xyz and rpy.
    :param xyz: Translation vector [x, y, z].
    :param rpy: Rotation vector [roll, pitch, yaw] in radians.
    :return: 4x4 transformation matrix.
    """
    rotation = R.from_euler('xyz', rpy).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = xyz
    return T

def visualize_gripper(T_global, base_link_color=[1, 0.5, 1]):
    gripper = []
    transformations = {
        "base_link_collision": create_transformation_matrix([0, 0, -0.06], [0, 0, 0]),
        "left_outer_knuckle": create_transformation_matrix([2.9948E-14, 0.021559, 0.015181], [0, 0, 0]),
        "left_finger": create_transformation_matrix([-2.4536E-14, -0.016413, 0.029258], [0, 0, 0]),
        "left_inner_knuckle": create_transformation_matrix([1.86600784687907E-06, 0.0220467847633621, 0.0261334672830885], [0, 0, 0]),
        "right_outer_knuckle": create_transformation_matrix([-3.1669E-14, -0.021559, 0.015181], [0, 0, 0]),
        "right_inner_knuckle": create_transformation_matrix([1.866E-06, -0.022047, 0.026133], [0, 0, 0]),
        "right_finger": create_transformation_matrix([2.5618E-14, 0.016413, 0.029258], [0, 0, 0]),
    }
    # Directory containing the robot arm meshes
    mesh_dir = '/home/u-ril/gaze2grasp/meshes'
    # Load meshes, apply transformations, and set colors
    for part, T_local in transformations.items():
        mesh_path = os.path.join(mesh_dir, f"{part}.stl")
        if os.path.exists(mesh_path):
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            mesh.transform(T_local)
            mesh.transform(T_global)
            if part == "base_link_collision":
                mesh.paint_uniform_color(base_link_color)  
            else:
                mesh.paint_uniform_color([0, 0, 0])  # Black for other parts
            gripper.append(mesh)
    return gripper