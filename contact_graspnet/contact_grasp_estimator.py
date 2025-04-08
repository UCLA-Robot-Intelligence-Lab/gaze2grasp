import importlib
import numpy as np
import sys
import os
import open3d as o3d
import time
import copy
from scipy.spatial.transform import Rotation 

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
TF2 = True

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, 'pointnet2',  'utils'))
sys.path.append(os.path.abspath(__file__))

from tf_train_ops import get_bn_decay
import config_utils
from data import farthest_points, distance_by_translation_point, preprocess_pc_for_inference, regularize_pc_point_count, depth2pc, reject_median_outliers

import open3d as o3d
import numpy as np

def visualize_zxy_planes_actual_values(pcd):
    geometries = [pcd]
    colors = [
        [1, 0, 0],  # Red
        [0, 1, 0],  # Green
        [0, 0, 1],  # Blue
        [1, 1, 0],  # Yellow
        [1, 0, 1]   # Magenta
    ]

    plane_size = 2.0  # Make planes larger for better visibility
    half_size = plane_size / 2
    plane_thickness = 0.01

    '''# Z-Planes (no rotation needed)
    for i, z in enumerate([-0.15, 0, 0.1]):
        plane = o3d.geometry.TriangleMesh.create_box(
            width=plane_size, 
            height=plane_size, 
            depth=plane_thickness
        )
        plane.translate((-half_size, -half_size, z - plane_thickness/2))
        plane.paint_uniform_color(colors[i % len(colors)])
        geometries.append(plane)'''

    # X-Planes (rotated around Y-axis)
    for i, x in enumerate([0,0.55]):
        plane = o3d.geometry.TriangleMesh.create_box(
            width=plane_size, 
            height=plane_size, 
            depth=plane_thickness
        )
        # Move to origin, rotate, then move to desired x-position
        plane.translate((-half_size, -half_size, -plane_thickness/2))
        plane.rotate(plane.get_rotation_matrix_from_xyz((0, np.pi/2, 0)))
        plane.translate((x, 0, 0))
        plane.paint_uniform_color(colors[i % len(colors)])
        geometries.append(plane)

    # Y-Planes (rotated around X-axis)
    for i, y in enumerate([-0.55, 0.55]):
        plane = o3d.geometry.TriangleMesh.create_box(
            width=plane_size, 
            height=plane_size, 
            depth=plane_thickness
        )
        # Move to origin, rotate, then move to desired y-position
        plane.translate((-half_size, -half_size, -plane_thickness/2))
        plane.rotate(plane.get_rotation_matrix_from_xyz((-np.pi/2, 0, 0)))
        plane.translate((0, y, 0))
        plane.paint_uniform_color(colors[i % len(colors)])
        geometries.append(plane)

    # Add coordinate axes
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    geometries.append(origin)

    o3d.visualization.draw_geometries(geometries)
class GraspEstimator:
    """
    Class for building and inferencing Contact-GraspNet
    
    :param cfg: config dict
    """
    def __init__(self, cfg):
        
        if 'surface_grasp_logdir_folder' in cfg:
            # for sim evaluation
            self._contact_grasp_cfg = config_utils.load_config(cfg['surface_grasp_logdir_folder'], batch_size=1, arg_configs=cfg.arg_configs)
            self._cfg = cfg
            self._num_samples = self._cfg.num_samples
        else:
            self._contact_grasp_cfg = cfg

        self._model_func = importlib.import_module(self._contact_grasp_cfg['MODEL']['model'])
        self._num_input_points = self._contact_grasp_cfg['DATA']['raw_num_points'] if 'raw_num_points' in self._contact_grasp_cfg['DATA'] else self._contact_grasp_cfg['DATA']['num_point']
        
        self.placeholders = self._model_func.placeholder_inputs(self._contact_grasp_cfg['OPTIMIZER']['batch_size'],
                                                                self._num_input_points, 
                                                                self._contact_grasp_cfg['DATA']['input_normals'])
        self.model_ops = {}

    def build_network(self):
        """
        Build tensorflow graph and grasp representation
        :returns: tensorflow ops to infer the model predictions
        """
        global_config = self._contact_grasp_cfg

        # Note the global_step=step parameter to minimize. 
        # That tells the optimizer to helpfully increment the 'step' parameter for you every time it trains.
        step = tf.Variable(0)
        bn_decay = get_bn_decay(step, global_config['OPTIMIZER'])
        tf.summary.scalar('bn_decay', bn_decay)

        print("--- Get model")
        # Get model
        with tf.device('/GPU:0'):
            end_points = self._model_func.get_model(self.placeholders['pointclouds_pl'], self.placeholders['is_training_pl'], global_config, bn_decay=bn_decay)

            tf_bin_vals = self._model_func.get_bin_vals(global_config)
            offset_bin_pred_vals = tf.gather_nd(tf_bin_vals, tf.expand_dims(tf.argmax(end_points['grasp_offset_head'], axis=2), axis=2)) if global_config['MODEL']['bin_offsets'] else end_points['grasp_offset_pred'][:,:,0]

            grasp_preds = self._model_func.build_6d_grasp(end_points['approach_dir_head'], end_points['grasp_dir_head'], end_points['pred_points'], offset_bin_pred_vals, use_tf=True) # b x num_point x 4 x 4
    
        self.model_ops = {'pointclouds_pl': self.placeholders['pointclouds_pl'],
                    'cam_poses_pl': self.placeholders['cam_poses_pl'],
                    'scene_idx_pl': self.placeholders['scene_idx_pl'],
                    'is_training_pl': self.placeholders['is_training_pl'],
                    'grasp_dir_pred': end_points['grasp_dir_head'],
                    'binary_seg_head': end_points['binary_seg_head'],
                    'binary_seg_pred': end_points['binary_seg_pred'],
                    'grasp_offset_head': end_points['grasp_offset_head'],
                    'grasp_offset_pred': end_points['grasp_offset_pred'],
                    'approach_dir_pred': end_points['approach_dir_head'],
                    'pred_points': end_points['pred_points'],
                    'offset_pred_idcs_pc': tf.argmax(end_points['grasp_offset_head'], axis=2) if global_config['MODEL']['bin_offsets'] else None,
                    'offset_bin_pred_vals': offset_bin_pred_vals,
                    'grasp_preds': grasp_preds,
                    'step': step,
                    'end_points': end_points}

        self.inference_ops = [self.model_ops['grasp_preds'], self.model_ops['binary_seg_pred'], self.model_ops['pred_points']]
        if self.model_ops['offset_bin_pred_vals'] is None:
            self.inference_ops.append(self.model_ops['grasp_offset_head'])
        else:
            self.inference_ops.append(self.model_ops['offset_bin_pred_vals'])

        return self.model_ops
        
    def load_weights(self, sess, saver, log_dir, mode='test'):
        """
        Load checkpoint weights
        :param sess: tf.Session
        :param saver: tf.train.Saver        
        """
        
        chkpt = tf.train.get_checkpoint_state(log_dir)
        if chkpt and chkpt.model_checkpoint_path:
            print(('loading ',  chkpt.model_checkpoint_path))
            saver.restore(sess, chkpt.model_checkpoint_path)
        else:
            if mode == 'test':
                print('no checkpoint in ', log_dir)
                exit()
            else:
                sess.run(tf.global_variables_initializer())

    def filter_segment(self, contact_pts, segment_pc, thres=0.00001):
        """
        Filter grasps to obtain contacts on specified point cloud segment
        
        :param contact_pts: Nx3 contact points of all grasps in the scene
        :param segment_pc: Mx3 segmented point cloud of the object of interest
        :param thres: maximum distance in m of filtered contact points from segmented point cloud
        :returns: Contact/Grasp indices that lie in the point cloud segment
        """
        filtered_grasp_idcs = np.array([],dtype=np.int32)
        
        if contact_pts.shape[0] > 0 and segment_pc.shape[0] > 0:
            try:
                dists = contact_pts[:,:3].reshape(-1,1,3) - segment_pc.reshape(1,-1,3)           
                min_dists = np.min(np.linalg.norm(dists,axis=2),axis=1)
                filtered_grasp_idcs = np.where(min_dists<thres)
            except:
                pass
            
        return filtered_grasp_idcs

    def extract_3d_cam_boxes(self, full_pc, pc_segments, min_size=0.2, max_size=0.3, max_z_height=0.45):
        """
        Extract 3D bounding boxes with a maximum z-height constraint.

        :param full_pc: Nx3 scene point cloud
        :param pc_segments: Dictionary of segmented point clouds
        :param min_size: minimum side length of the 3D bounding box
        :param max_size: maximum side length of the 3D bounding box
        :param max_z_height: maximum z-height of the upper surface of the box
        :returns: (pc_regions, obj_centers) Point cloud box regions and their centers
        """

        pc_regions = {}
        obj_centers = {}
        geometries = []  # List to hold all geometries for visualization

        # Convert full_pc to Open3D PointCloud for visualization
        full_pcd = o3d.geometry.PointCloud()
        full_pcd.points = o3d.utility.Vector3dVector(full_pc)
        geometries.append(full_pcd)

        for i in pc_segments:
            pc_segments[i] = reject_median_outliers(pc_segments[i], m=0.4, z_only=False)

            if np.any(pc_segments[i]):
                max_bounds = np.max(pc_segments[i][:, :3], axis=0)
                min_bounds = np.min(pc_segments[i][:, :3], axis=0)

                obj_extent = max_bounds - min_bounds
                box_size = np.max(obj_extent)  # Use actual extent, not doubled

                # Apply size constraints
                box_size = np.clip(box_size, min_size, max_size)

                # Calculate center with z-offset
                obj_center = (max_bounds + min_bounds) / 2
                obj_center[2] -= 0.06  # Apply vertical offset

                # Calculate bounds with precise alignment
                lower_bound = obj_center - box_size / 2
                upper_bound = obj_center + box_size / 2
                #top_z = upper_bound[2]
                #print(f"Z-coordinate of top of box for object {i}: {top_z}")
                # Apply maximum z-height constraint
                upper_bound[2] = min(upper_bound[2], lower_bound[2] + max_z_height)

                condition = np.all(full_pc > lower_bound, axis=1) & np.all(full_pc < upper_bound, axis=1)
                print(f"Number of points in full_pc satisfying condition for object {i}:", np.sum(condition))

                partial_pc = full_pc[condition]

                if np.any(partial_pc):
                    partial_pc = regularize_pc_point_count(partial_pc, self._contact_grasp_cfg['DATA']['raw_num_points'], use_farthest_point=self._contact_grasp_cfg['DATA']['use_farthest_point'])
                    pc_regions[i] = partial_pc
                    obj_centers[i] = obj_center

                    # Create and add bounding box to visualization
                    box = o3d.geometry.AxisAlignedBoundingBox(min_bound=lower_bound, max_bound=upper_bound)
                    box.color = (1, 0, 0)  # Red box
                    geometries.append(box)

                else:
                    print(f"No points in full_pc satisfy the condition for object {i}")
            else:
                print(f"pc_segments[{i}] is empty")

        # Visualize all geometries (point cloud and boxes)
        o3d.visualization.draw_geometries(geometries)

        return pc_regions, obj_centers

    def predict_grasps(self, sess, pc, constant_offset=False, convert_cam_coords=True, forward_passes=1):
        """
        Predict raw grasps on point cloud

        :param sess: tf.Session_contact_grasp_cfgs
        """
        
        # Convert point cloud coordinates from OpenCV to internal coordinates (x left, y up, z front)
        pc, pc_mean = preprocess_pc_for_inference(pc.squeeze(), self._num_input_points, return_mean=True, convert_to_internal_coords=convert_cam_coords)

        if len(pc.shape) == 2:
            pc_batch = pc[np.newaxis,:,:]

        if forward_passes > 1:
            pc_batch = np.tile(pc_batch, (forward_passes,1,1))
            
        feed_dict = {self.placeholders['pointclouds_pl']: pc_batch,
                    self.placeholders['is_training_pl']: False}
        
        with tf.device('/GPU:0'): 
            # Run model inference
            pred_grasps_cam, pred_scores, pred_points, offset_pred = sess.run(self.inference_ops, feed_dict=feed_dict)

        pred_grasps_cam = pred_grasps_cam.reshape(-1, *pred_grasps_cam.shape[-2:])
        pred_points = pred_points.reshape(-1, pred_points.shape[-1])
        pred_scores = pred_scores.reshape(-1)
        offset_pred = offset_pred.reshape(-1)
        
        # uncenter grasps
        pred_grasps_cam[:,:3, 3] += pc_mean.reshape(-1,3)
        pred_points[:,:3] += pc_mean.reshape(-1,3)

        if constant_offset:
            offset_pred = np.array([[self._contact_grasp_cfg['DATA']['gripper_width']-self._contact_grasp_cfg['TEST']['extra_opening']]*self._contact_grasp_cfg['DATA']['num_point']])
        #print('extra opening n gripper width: ', self._contact_grasp_cfg['TEST']['extra_opening'], self._contact_grasp_cfg['DATA']['gripper_width'])
        gripper_openings = np.minimum(offset_pred + self._contact_grasp_cfg['TEST']['extra_opening'], self._contact_grasp_cfg['DATA']['gripper_width'])
        #print("GRIPPER OPENING", gripper_openings)
        with_replacement = self._contact_grasp_cfg['TEST']['with_replacement'] if 'with_replacement' in self._contact_grasp_cfg['TEST'] else False
        
        selection_idcs = self.select_grasps(pred_points[:,:3], pred_scores, 
                                            self._contact_grasp_cfg['TEST']['max_farthest_points'], 
                                            self._contact_grasp_cfg['TEST']['num_samples'], 
                                            self._contact_grasp_cfg['TEST']['first_thres'], 
                                            self._contact_grasp_cfg['TEST']['second_thres'] if 'second_thres' in self._contact_grasp_cfg['TEST'] else self._contact_grasp_cfg['TEST']['first_thres'], 
                                            with_replacement=self._contact_grasp_cfg['TEST']['with_replacement'])

        if not np.any(selection_idcs):
            selection_idcs=np.array([], dtype=np.int32)
        #print("CENTRE TO TIP", self._contact_grasp_cfg['TEST']['center_to_tip'])
        
        if 'center_to_tip' in self._contact_grasp_cfg['TEST'] and self._contact_grasp_cfg['TEST']['center_to_tip']:
            #print('pred_grasps_cam:', pred_grasps_cam)
            pred_grasps_cam[:,:3, 3] -= pred_grasps_cam[:,:3,2]*(self._contact_grasp_cfg['TEST']['center_to_tip']/2)
            #print('pred_grasps_cam after centre to tip:', pred_grasps_cam)
        
        # convert back to opencv coordinates
        if convert_cam_coords:
            pred_grasps_cam[:,:2, :] *= -1
            pred_points[:,:2] *= -1

        return pred_grasps_cam[selection_idcs], pred_scores[selection_idcs], pred_points[selection_idcs].squeeze(), gripper_openings[selection_idcs].squeeze()
   
    def predict_scene_grasps(self, sess, pc_full, pc_segments={}, local_regions=False, filter_grasps=False, forward_passes=1):
        """
        Predict num_point grasps on a full point cloud or in local box regions around point cloud segments.

        Arguments:
            sess {tf.Session} -- Tensorflow Session
            pc_full {np.ndarray} -- Nx3 full scene point cloud  

        Keyword Arguments:
            pc_segments {dict[int, np.ndarray]} -- Dict of Mx3 segmented point clouds of objects of interest (default: {{}})
            local_regions {bool} -- crop 3D local regions around object segments for prediction (default: {False})
            filter_grasps {bool} -- filter grasp contacts such that they only lie within object segments (default: {False})
            forward_passes {int} -- Number of forward passes to run on each point cloud. (default: {1})

        Returns:
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray] -- pred_grasps_cam, scores, contact_pts, gripper_openings
        """

        pred_grasps_cam, scores, contact_pts, gripper_openings = {}, {}, {}, {}

        # Predict grasps in local regions or full pc
        if local_regions:
            pc_regions, _ = self.extract_3d_cam_boxes(pc_full, pc_segments)

            for k, pc_region in pc_regions.items():
                pred_grasps_cam[k], scores[k], contact_pts[k], gripper_openings[k] = self.predict_grasps(sess, pc_region, convert_cam_coords=False, forward_passes=forward_passes)
            print('Generated {} grasps in local_regions'.format(len(pred_grasps_cam[True])))

        else:
            print('local_regions is False. Regularizing pc_full to {} points'.format(self._contact_grasp_cfg['DATA']['raw_num_points']))
            pc_full = regularize_pc_point_count(pc_full, self._contact_grasp_cfg['DATA']['raw_num_points'])
            print('Predicting grasps on full scene point cloud...')
            pred_grasps_cam[-1], scores[-1], contact_pts[-1], gripper_openings[-1] = self.predict_grasps(sess, pc_full, convert_cam_coords=True, forward_passes=forward_passes)
            print('Generated {} grasps'.format(len(pred_grasps_cam[-1])))

        # Filter grasp contacts to lie within object segment
        if filter_grasps:
            segment_keys = contact_pts.keys() if local_regions else pc_segments.keys()
            for k in segment_keys:
                j = k if local_regions else -1
                if np.any(pc_segments[k]) and np.any(contact_pts[j]):
                    segment_idcs = self.filter_segment(contact_pts[j], pc_segments[k], thres=self._contact_grasp_cfg['TEST']['filter_thres'])
                    print('segment_idcs: ', segment_idcs)
                    pred_grasps_cam[k] = pred_grasps_cam[j][segment_idcs]
                    scores[k] = scores[j][segment_idcs]
                    contact_pts[k] = contact_pts[j][segment_idcs]
                    try:
                        gripper_openings[k] = gripper_openings[j][segment_idcs]
                    except:
                        print('skipped gripper openings {}'.format(gripper_openings[j]))

                    if local_regions and np.any(pred_grasps_cam[k]):
                        print('Generated {} grasps for object {}'.format(len(pred_grasps_cam[k]), k))
                else:
                    print('skipping obj {} since  np.any(pc_segments[k]) {} and np.any(contact_pts[j]) is {}'.format(k, np.any(pc_segments[k]), np.any(contact_pts[j])))

            if not local_regions:
                del pred_grasps_cam[-1], scores[-1], contact_pts[-1], gripper_openings[-1]

        return pred_grasps_cam, scores, contact_pts, gripper_openings
    
    def select_grasps(self, contact_pts, contact_conf, max_farthest_points = 150, num_grasps = 200, first_thres = 0.25, second_thres = 0.2, with_replacement=False):
        """
        Select subset of num_grasps by contact confidence thresholds and farthest contact point sampling. 

        1.) Samples max_farthest_points among grasp contacts with conf > first_thres
        2.) Fills up remaining grasp contacts to a maximum of num_grasps with highest confidence contacts with conf > second_thres
        
        Arguments:
            contact_pts {np.ndarray} -- num_point x 3 subset of input point cloud for which we have predictions 
            contact_conf {[type]} -- num_point x 1 confidence of the points being a stable grasp contact

        Keyword Arguments:
            max_farthest_points {int} -- Maximum amount from num_grasps sampled with farthest point sampling (default: {150})
            num_grasps {int} -- Maximum number of grasp proposals to select (default: {200})
            first_thres {float} -- first confidence threshold for farthest point sampling (default: {0.6})
            second_thres {float} -- second confidence threshold for filling up grasp proposals (default: {0.6})
            with_replacement {bool} -- Return fixed number of num_grasps with conf > first_thres and repeat if there are not enough (default: {False})

        Returns:
            [np.ndarray] -- Indices of selected contact_pts 
        """

        grasp_conf = contact_conf.squeeze()
        contact_pts = contact_pts.squeeze()

        conf_idcs_greater_than = np.nonzero(grasp_conf > first_thres)[0]
        _, center_indexes = farthest_points(contact_pts[conf_idcs_greater_than,:3], np.minimum(max_farthest_points, len(conf_idcs_greater_than)), distance_by_translation_point, return_center_indexes = True)

        remaining_confidences = np.setdiff1d(np.arange(len(grasp_conf)), conf_idcs_greater_than[center_indexes])
        sorted_confidences = np.argsort(grasp_conf)[::-1]
        mask = np.in1d(sorted_confidences, remaining_confidences)
        sorted_remaining_confidence_idcs = sorted_confidences[mask]
        
        if with_replacement:
            selection_idcs = list(conf_idcs_greater_than[center_indexes])
            j=len(selection_idcs)
            while j < num_grasps and conf_idcs_greater_than.shape[0] > 0:
                selection_idcs.append(conf_idcs_greater_than[j%len(conf_idcs_greater_than)])
                j+=1
            selection_idcs = np.array(selection_idcs)

        else:
            remaining_idcs = sorted_remaining_confidence_idcs[:num_grasps-len(conf_idcs_greater_than[center_indexes])]
            remaining_conf_idcs_greater_than = np.nonzero(grasp_conf[remaining_idcs] > second_thres)[0]
            selection_idcs = np.union1d(conf_idcs_greater_than[center_indexes], remaining_idcs[remaining_conf_idcs_greater_than])
        return selection_idcs

    def extract_point_clouds(self, depth, K, segmap=None, rgb=None, z_range=[0.2,1.3], segmap_id=0, skip_border_objects=False, margin_px=5):
        """
        Converts depth map + intrinsics to point cloud. 
        If segmap is given, also returns segmented point clouds. If rgb is given, also returns pc_colors.

        Arguments:
            depth {np.ndarray} -- HxW depth map in m
            K {np.ndarray} -- 3x3 camera Matrix

        Keyword Arguments:
            segmap {np.ndarray} -- HxW integer array that describes segeents (default: {None})
            rgb {np.ndarray} -- HxW rgb image (default: {None})
            z_range {list} -- Clip point cloud at minimum/maximum z distance (default: {[0.2,1.8]})
            segmap_id {int} -- Only return point cloud segment for the defined id (default: {0})
            skip_border_objects {bool} -- Skip segments that are at the border of the depth map to avoid artificial edges (default: {False})
            margin_px {int} -- Pixel margin of skip_border_objects (default: {5})

        Returns:
            [np.ndarray, dict[int:np.ndarray], np.ndarray] -- Full point cloud, point cloud segments, point cloud colors
        """

        if K is None:
            raise ValueError('K is required either as argument --K or from the input numpy file')
                
        # Convert to point cloud
        pc_full, pc_colors = depth2pc(depth, K, rgb)

        # Threshold distance
        if pc_colors is not None:
            pc_colors = pc_colors[(pc_full[:, 2] < z_range[1]) & (pc_full[:, 2] > z_range[0])] 
        pc_full = pc_full[(pc_full[:, 2] < z_range[1]) & (pc_full[:, 2] > z_range[0])]
            
        # Extract instance point clouds from segmap and depth map
        pc_segments = {}
        if segmap is not None:
            obj_instances = [segmap_id] if segmap_id else np.unique(segmap[segmap > 0])
            #print("Object instances:", obj_instances)  # Debugging print

            for i in obj_instances:
                if skip_border_objects and not i == segmap_id:
                    obj_i_y, obj_i_x = np.where(segmap == i)
                    if np.any(obj_i_x < margin_px) or np.any(obj_i_x > segmap.shape[1] - margin_px) or np.any(obj_i_y < margin_px) or np.any(obj_i_y > segmap.shape[0] - margin_px):
                        #print('Object {} not entirely in image bounds, skipping'.format(i))
                        continue

                inst_mask = segmap == i
                #print(f"Instance {i} mask sum:", np.sum(inst_mask))
                #print("Inst mask:", inst_mask)
                #print("Inst mask dtype:", inst_mask.dtype)

                pc_segment, _ = depth2pc(depth * inst_mask, K)
                #print(f"Instance {i} point cloud shape before filtering:", pc_segment.shape)
                #print(f"Instance {i} point cloud dtype:", pc_segment.dtype)
                #print(f"pc_segment:", pc_segment)
                pc_segment = pc_segment[(pc_segment[:, 2] < z_range[1]) & (pc_segment[:, 2] > z_range[0])]
                #print(f"Instance {i} point cloud shape after filtering:", pc_segment.shape)  # Debugging print

                pc_segments[i] = pc_segment

        return pc_full, pc_segments, pc_colors
            
    def predict_scene_grasps_from_depth_K_and_2d_seg(self, sess, depth, segmap, K, TCR, z_range=[0.2,1.3], local_regions=False, filter_grasps=False, segmap_id=0, skip_border_objects=False, margin_px=5, rgb=None, forward_passes=1):
        """ Combines converting to point cloud(s) and predicting scene grasps into one function """
        if len(depth.shape) == 2 and len(segmap.shape) == 2 and len(K.shape) == 2 and len(TCR.shape) == 2:
            #print(depth.shape)
            ##print(K.shape)
            #print(rgb.shape)
            #print(segmap.shape)
            pc_full, pc_segments, pc_color  = self.extract_point_clouds(depth, K, segmap=segmap, segmap_id=segmap_id, skip_border_objects=skip_border_objects, margin_px=margin_px, z_range=z_range, rgb=rgb[:, :, ::-1])
            #print(np.array(pc_full).shape)
            #print(np.array(pc_color).shape)
            #print(np.array(pc_segments[True]).shape)

            points_3d_homog = np.vstack((np.array(pc_full).T, np.ones((1, np.array(pc_full).shape[0]))))  # Shape: (4, N)
            points_3d_transformed = (TCR @ points_3d_homog).T
            pc_segments_homg = np.vstack((np.array(pc_segments[True]).T, np.ones((1, np.array(pc_segments[True]).shape[0]))))  # Shape: (4, N)
            pc_segments[True] = (TCR @ pc_segments_homg).T
            pred_grasps_cam, scores, contact_pts, gripper_openings = self.predict_scene_grasps(sess, points_3d_transformed, pc_segments, local_regions=local_regions, filter_grasps=filter_grasps, forward_passes=forward_passes)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_3d_transformed)
            pcd.colors = o3d.utility.Vector3dVector(pc_color/255)
            #R = TCR[:3, :3]  # Extract rotation
            #t = TCR[:3, 3]   # Extract translation
            #R = TCR[:3, :3]  # 3x3 rotation
            #t = TCR[:3, 3]   # 3x1 translation

            #R_inv = R.T      # Rotation inverse (transpose)
            #t_inv = -R_inv @ t  # Translation inverse

            #for grasp in pred_grasps_cam[True]:
                # Apply inverse rotation and translation (optimized)
            #    grasp[:3, 3] = R_inv @ grasp[:3, 3] + t_inv  # p = R⁻¹ p' + t_inv
                
                # Apply inverse rotation to orientation
            #    grasp[:3, :3] = R_inv @ grasp[:3, :3]
            return pcd, pcd, pred_grasps_cam, scores, contact_pts, gripper_openings
        elif len(depth.shape) == 3 and len(segmap.shape) == 3 and len(K.shape) == 3 and len(TCR.shape) == 3:
            print("MERGING POINT CLOUDS FOR CONTACT_GRASPNET_PRED")
            merged_segments, pcds= {}, []
            for i in range(K.shape[0]):
                #print(depth[i].shape)
                #print(K[i].shape)
                #print(rgb[i].shape)
                #print(segmap[i].shape)
                pc_full, pc_segments, pc_color  = self.extract_point_clouds(depth[i], K[i], segmap=segmap[i], segmap_id=segmap_id, skip_border_objects=skip_border_objects, margin_px=margin_px, z_range=z_range, rgb=rgb[i][:, :, ::-1])
                
                print(np.array(pc_full).shape)
                print(np.array(pc_color).shape)            
                print(np.array(pc_segments[True]).shape)


                points_3d_homog = np.vstack((np.array(pc_full).T, np.ones((1, np.array(pc_full).shape[0]))))  # Shape: (4, N)
                points_3d_transformed = (TCR[i] @ points_3d_homog).T
                pc_segments_homg = np.vstack((np.array(pc_segments[True]).T, np.ones((1, np.array(pc_segments[True]).shape[0]))))  # Shape: (4, N)
                pc_segments[True] = (TCR[i] @ pc_segments_homg).T
                for key, segment in pc_segments.items():
                    if key in merged_segments:
                        merged_segments[key] = np.concatenate((merged_segments[key], segment), axis=0)
                    else:
                        merged_segments[key] = segment
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_3d_transformed)
                pcd.colors = o3d.utility.Vector3dVector(pc_color/255)
                #o3d.io.write_point_cloud(f"{i}.pcd", pcd)
                pcds.append(pcd)
            pcd_combined = o3d.geometry.PointCloud()
            for pcd in pcds:
                pcd_combined += pcd
            
            # Example usage (replace pcd_combined with your point cloud)
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            visualize_zxy_planes_actual_values(pcd_combined)
            vis.destroy_window()
            #o3d.io.write_point_cloud("combined_pcd.pcd", pcd_combined)
            #rot = Rotation.from_euler('xy',[90,-90], degrees=True)
            rot = Rotation.from_euler('x', 120, degrees=True)
            rot1 = Rotation.from_euler('x', 180, degrees=True)
            rot2 = Rotation.from_euler('x', -120, degrees=True)  # Example third rotation

            # Calculate inverse rotation matrices
            rot_inv_matrix = rot.inv().as_matrix()
            rot1_inv_matrix = rot1.inv().as_matrix()
            rot2_inv_matrix = rot2.inv().as_matrix()

            # Rotate point clouds and segments
            pc_temp = (rot.as_matrix() @ np.asarray(pcd_combined.points).T).T
            merged_segments_temp = (rot.as_matrix() @ merged_segments[True].T).T
            merged_segments_rotated = {True: merged_segments_temp}

            pc_temp1 = (rot1.as_matrix() @ np.asarray(pcd_combined.points).T).T
            merged_segments_temp1 = (rot1.as_matrix() @ merged_segments[True].T).T
            merged_segments_rotated1 = {True: merged_segments_temp1}

            pc_temp2 = (rot2.as_matrix() @ np.asarray(pcd_combined.points).T).T
            merged_segments_temp2 = (rot2.as_matrix() @ merged_segments[True].T).T
            merged_segments_rotated2 = {True: merged_segments_temp2}


            # Visualize (optional, for debugging)
            pcd_temp = o3d.geometry.PointCloud()
            pcd_temp.points = o3d.utility.Vector3dVector(pc_temp)
            pcd_temp1 = o3d.geometry.PointCloud()
            pcd_temp1.points = o3d.utility.Vector3dVector(pc_temp1)
            pcd_temp2 = o3d.geometry.PointCloud()
            pcd_temp2.points = o3d.utility.Vector3dVector(pc_temp2)
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(pcd_combined)
            vis.add_geometry(pcd_temp)
            vis.add_geometry(pcd_temp1)
            vis.add_geometry(pcd_temp2)
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            vis.add_geometry(origin)
            vis.run()
            vis.destroy_window()

            # Predict grasps in the rotated frames
            pred_grasps_cam, scores, contact_pts, gripper_openings = self.predict_scene_grasps(
                sess, pc_temp, merged_segments_rotated, local_regions=local_regions, filter_grasps=filter_grasps, forward_passes=forward_passes
            )
            pred_grasps_cam1, scores1, contact_pts1, gripper_openings1 = self.predict_scene_grasps(
                sess, pc_temp1, merged_segments_rotated1, local_regions=local_regions, filter_grasps=filter_grasps, forward_passes=forward_passes
            )
            pred_grasps_cam2, scores2, contact_pts2, gripper_openings2 = self.predict_scene_grasps(
                sess, pc_temp2, merged_segments_rotated2, local_regions=local_regions, filter_grasps=filter_grasps, forward_passes=forward_passes
            )

            # Efficiently un-rotate the predicted grasps
            grasps1 = pred_grasps_cam[True]
            grasps2 = pred_grasps_cam1[True]
            grasps3 = pred_grasps_cam2[True]

            rotated_grasps1 = grasps1.copy()
            rotated_grasps1[:, :3, :3] = np.einsum('ij,ajk->aik', rot_inv_matrix, grasps1[:, :3, :3])
            rotated_grasps1[:, :3, 3] = np.einsum('ij,aj->ai', rot_inv_matrix, grasps1[:, :3, 3])

            rotated_grasps2 = grasps2.copy()
            rotated_grasps2[:, :3, :3] = np.einsum('ij,ajk->aik', rot1_inv_matrix, grasps2[:, :3, :3])
            rotated_grasps2[:, :3, 3] = np.einsum('ij,aj->ai', rot1_inv_matrix, grasps2[:, :3, 3])

            rotated_grasps3 = grasps3.copy()
            rotated_grasps3[:, :3, :3] = np.einsum('ij,ajk->aik', rot2_inv_matrix, grasps3[:, :3, :3])
            rotated_grasps3[:, :3, 3] = np.einsum('ij,aj->ai', rot2_inv_matrix, grasps3[:, :3, 3])

            all_rotated_grasps = np.concatenate([rotated_grasps1, rotated_grasps2, rotated_grasps3])
            pred_grasps_cam[True] = all_rotated_grasps

            # Un-rotate scores
            all_scores = np.concatenate([scores[True], scores1[True], scores2[True]])

            # Efficiently un-rotate contact points
            contact_pts1_array = contact_pts[True]
            contact_pts2_array = contact_pts1_array.copy()
            contact_pts3_array = contact_pts1_array.copy() # Assuming you want to un-rotate the same original contact_pts by all three inverse rotations

            rotated_contact_pts1 = np.einsum('ij,aj->ai', rot_inv_matrix, contact_pts1_array)
            rotated_contact_pts2 = np.einsum('ij,aj->ai', rot1_inv_matrix, contact_pts2_array)
            rotated_contact_pts3 = np.einsum('ij,aj->ai', rot2_inv_matrix, contact_pts3_array)

            all_rotated_contact_pts = np.concatenate([rotated_contact_pts1, rotated_contact_pts2, rotated_contact_pts3])

            # Un-rotate gripper openings
            all_gripper_openings = np.concatenate([gripper_openings[True], gripper_openings1[True], gripper_openings2[True]])
            #print("Grasp after rotation:", pred_grasps_cam[True][0])
            #pred_grasps_cam, scores, contact_pts, gripper_openings = self.predict_scene_grasps(sess, np.asarray(pcd_combined.points), merged_segments, local_regions=local_regions, filter_grasps=filter_grasps, forward_passes=forward_passes)
            
            return pcd_combined, pcds, pred_grasps_cam, all_scores, all_rotated_contact_pts, all_gripper_openings
        else:
            print("INVALID K MATRIX INPUT")
            return None
        #print(pc_full)
        #print(pc_segments)
        #pred_grasps_cam = self.predict_scene_grasps(sess, pc_full, pc_segments=None, local_regions=None, filter_grasps=False, forward_passes=1)[0] 
        #print('INTERMEDIARY: Generated {} grasps'.format(len(pred_grasps_cam[-1])))
