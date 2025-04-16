
# 📦 Gaze2Grasp: Live Visualization & Data Collection Guide

This document outlines how to use `live_visualization.py` for collecting grasp data using gaze input and robot control, along with common troubleshooting steps and folder output structure.

---

## ⚠️ Common Errors & Fixes

| Issue                             | Solution                                             |
|----------------------------------|------------------------------------------------------|
| **Realsense camera fails to start** | Replug the USB cables of the camera.                 |
| **XArm fails to start**            | Power cycle the XArm (turn it off and on).           |
| **No GPU detected on computer**   | Restart the computer to reset GPU availability.      |

---

## ▶️ How to Use `live_visualization.py`

1. **Start the script**:
   ```bash
   python -m visualizations.live_visualization 
   ```

2. **Interactive Annotation Flow**:
   - A window labeled **`81`** will appear:
     - **Double-click** on the object of interest.
     - Press `Esc`.
   - A window labeled **`56`** will then appear:
     - **Double-click** on the **same** object of interest.
     - **Double-click** again at the desired **placement location** for the object.
     - Press `Esc`.

3. **Saving the Data**:
   - After pressing `Esc`, you'll be prompted in the terminal to enter a folder name:
     - **Type a name** to save under a custom label.
     - **Press Enter** to use the default location: `default_images/`.

4. **Robot Execution**:
   - Once saved, the robot will **move to the last predicted grasp pose**.

5. **Folder Naming Tip**:
   - After reviewing the saved predictions, rename the folder as:
     ```
     object + correct color option
     ```
   - Example: `mug_red_graspable`

---

## 📂 Output Folder Structure (`vlm_images/`)

Each run creates the following structure:

- `pc_81/`: Grasp predictions visualized from camera 81.
- `pc_56/`: Grasp predictions visualized from camera 56.
- `pc_combined/`: Grasps from the merged point cloud (multi-view).
- `grasp_data/`: Numerical grasp predictions and metadata.
- `*_diagram.png`: Diagram summarizing all predicted grasps.
- `*_gaze_view.png`: Camera view with user gaze location marked.

---

## 📸 Data Collection Protocol

### ✅ Preparation
1. **Object Placement**:
   - Place the **object directly in front of the robot arm**.
   - Use a **small platform** to elevate the object for better grasp estimation.

2. **Scene Diversity**:
   - Try different **backgrounds, lighting, and object arrangements** to diversify your dataset.

---

### ✅ Running the Script
```bash
python -m visualizations.live_visualization 
```

Follow the instructions above to annotate the object and placement.

---

### ✅ Post-Run Review
1. Inspect the generated images and predicted grasps.
2. Decide if the grasp predictions are **usable for VLM training**.
3. If they are meaningful, **annotate the folder** with an appropriate name (e.g., `banana_green_side`).

---

Happy grasping! 🤖✋🧠
