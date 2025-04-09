# Gaze-Enabled Robotics Using Vision Language Models (GRUV)

This project combines gaze tracking, vision-language models (VLMs), and robotic grasping to create a shared autonomy system that enables gaze-based robot control. The system leverages multiple components such as SAM (Segment Anything Model), Contact-GraspNet, and VLM-guided grasp selection to achieve intuitive and efficient human-robot interaction.

---

## 🗂 Folder Structure

### `calib/`
- Contains point clouds, intrinsic/extrinsic parameters, and robot-to-camera transformations.
- `pc_calib_combined.py`: Combines calibration info to visualize point clouds from different views.

### `checkpoints/`
- Stores model checkpoints for **Contact-GraspNet**.

### `combined_images/`
- Contains images generated from running the full pipeline.

### `pointnet2/`
- Used for grasp generation via **Contact-GraspNet**.

### `meshes/`
- STL files for the xArm gripper and associated visualization.

### `tools/`
- *(TODO: Add description)*

### `ultralytics/`
- Utilized for the **Segment Anything Model (SAM)**.

### `vlm_images/`
- Stores visualized grasp samples generated for **VLM training**.

---

## ⚙️ Key Files and Modules

### `grasp.yml`
- Conda environment specifications.

### Top-Level Scripts
- `baseline.py`: Runs a gaze-controlled remote system. Uses gaze and ArUco segmentation to control the robot.
- `combined.py`: Executes the full gaze-to-robot-control pipeline.
- `multi_cam.py`: Defines the xArm setup and camera calibration to robot base.
- `rs_streamer.py`: Handles RealSense camera stream.
- `streaming_control.py`: Real-time gaze input controller to move the robot to stared-at points.

---

## 📦 Submodules

### `contact_graspnet/`
- `contact_grasp_estimator.py` updated with the following: 
  - Visualizes robot workspace in 3D.
  - Draws 3D camera boxes differently.
  - `predict_scene_grasps_from_depth_K_and_2d_seg()` modified to:
    - Accept multiple depth images.
    - Predict grasps from merged point clouds.
    - Output transformed point clouds.
    - Run grasp prediction from three different rotations for diverse options.
- `contact_graspnet.py`: Runs the main Contact-GraspNet inference.
- `common.py`, `config_utils.py`,`data.py`, `summaries.py`, `tf_train_ops.py`, `train.py`, `scene_renderer.py`: Contact-GraspNet functionality.

### `calib_utils/`, `vision_utils/`
- Utility functions.

### `gaze_model/`
- For inference of gaze from wearable device input.

### `segment/`
- Integrates **SAM** into the pipeline for segmentation tasks.

---

## 🧠 Supporting Modules

- `gaze_history.py`: 
  - Stores historical gaze data.
  - Outputs a boolean flag and gaze coordinates if the user is staring.
  
- `grasp_selector.py`: 
  - Uses clustering and distance metrics to select nearest and most distinct grasps.
  - Filters grasp predictions for **VLM** training and inference.

- `homography.py`: 
  - Transforms gaze points from ARIA to camera coordinates.

- `live_visualization.py`: 
  - Visualizes predicted grasps.
  - Uses gaze coordinates as input for visual output.

---

## 🚀 Pipeline Overview

1. **Gaze Inference**: Gaze data is captured and inferred using ARIA and processed via `gaze_model/`.
2. **Scene Segmentation**: SAM is used to segment the environment.
3. **Grasp Prediction**: Contact-GraspNet is used to generate 6-DoF grasp candidates from merged depth data.
4. **Grasp Selection**: Candidates are clustered and ranked using `grasp_selector.py`.
5. **VLM Integration**: Selected grasps are visualized and optionally used to train/infer with a vision-language model.
6. **Robot Control**: The robot moves based on the inferred gaze and selected grasp.

---

## 📝 TODOs
- [ ] Add detailed descriptions for `tools/`.
- [ ] Add visual diagrams of pipeline (optional).
- [ ] Link related papers or datasets if available.

---

## 🧪 Environment Setup

```bash
conda env create -f grasp.yml
conda activate grasp
