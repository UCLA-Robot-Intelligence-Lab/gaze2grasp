#import argparse
import ast
from segment.FastSAM.fastsam import FastSAMPrompt
from scipy.spatial import distance
import numpy as np

def run_fastsam_point_inference( #Use point prompt to get the mask
    model,
    point_prompt,
    input_img=None,
    imgsz=1024,
    iou=0.9,
    conf=0.4,
    point_label="[1,0]",
    device=None,
    retina=True,
):

    # Convert string inputs to actual Python lists
    point_label = ast.literal_eval(point_label)

    # Run inference
    results = model(
        input_img,
        device=device,
        retina_masks=retina,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
    )

    # Process prompts
    prompt_process = FastSAMPrompt(input_img, results, device=device)
    ann = prompt_process.point_prompt(points=point_prompt, pointlabel=point_label)
         
    return ann.reshape(ann.shape[1], ann.shape[2])

def select_from_sam_everything( #Segments everything and merge masks that is closest to the point prompt
    model,
    point_prompt,
    input_img=None,
    imgsz=1024,
    iou=0.9,
    conf=0.4,
    max_distance=100,
    device=None,
    retina=True,
):
    # Run inference
    results = model(
        input_img,
        device=device,
        retina_masks=retina,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
    )

    # Process prompts
    prompt_process = FastSAMPrompt(input_img, results, device=device)
    
    ann = prompt_process.everything_prompt()
    ann = ann.cpu().numpy()
    image_size = input_img.size[0] * input_img.size[1]
  
    filtered_masks = []
    if ann is not None:
        for mask in ann:
            mask_indices = np.argwhere(mask)
            if len(mask_indices) > 0:
                within_radius = False
                for y, x in mask_indices:
                    if distance.euclidean(point_prompt[0], (x, y)) <= max_distance:
                        within_radius = True
                        break
                if within_radius:
                    mask_size = np.count_nonzero(mask)
                    if mask_size <= 0.2* image_size:  # Check if mask is not larger than 50%
                        filtered_masks.append(mask)
    combined_mask = np.zeros((input_img.size[1], input_img.size[0]), dtype=np.uint8)
    if filtered_masks:
        for mask in filtered_masks:
            combined_mask = np.logical_or(combined_mask, mask)
    return combined_mask.astype(np.uint8)
