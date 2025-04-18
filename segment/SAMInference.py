#import argparse
import ast
from segment.FastSAM.fastsam import FastSAMPrompt
from scipy.spatial import distance
import numpy as np

'''
Run_fastsam_point_inference: Uses FastSAM’s point inference, with input as a list of gaze points 

Run_fastsam_single_point_inference: Uses FastSAM’s point inference, with input as a single gaze point

*Point inference was slow and gaze may not always be on the object for appropriate segmentation
-----------------------------------------
Select_from_sam_everything_points: Uses FastSAM’s everything inference, filters out masks if any part of it is outside a max distance or if the mask is too big 
(larger than 20% of scene), appends the centre of each masks

Select_from_sam_everything:  Uses FastSAM’s everything inference, filters out masks if any part of it is outside a max distance or if the mask is too big 
(larger than 20% of scene), merges those masks into a combined masks and appends the centre of each masks (option to include the largest mask in the filtered mask centers

*Everything inference is faster and centre of the masks are used to be transformed from aria/camera views for accuracy
'''

def run_fastsam_point_inference(
    model,
    point_prompts,  # Now accepts a list of point prompts
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
    print('point_label: ', point_label)
    '''
    original_height, original_width = input_img.shape[:2]
    resized_height, resized_width = imgsz, imgsz

    # Scale point prompts to the resized image coordinates
    scale_x = resized_width / original_width
    scale_y = resized_height / original_height
    scaled_point_prompts = [(int(point[0] * scale_x), int(point[1] * scale_y)) for point in point_prompts]
    '''
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

    combined_mask = np.zeros((input_img.shape[0], input_img.shape[1]), dtype=np.uint8)

    for point_prompt in point_prompts:  # Iterate through the list of point prompts
        print('point_prompt: ', point_prompt)
        ann = prompt_process.point_prompt(points=[point_prompt], pointlabel=point_label)
        mask = ann.reshape(ann.shape[1], ann.shape[2])
        combined_mask = np.logical_or(combined_mask, mask)  # Combine with previous masks
    return combined_mask  # Return the combined mask

def run_fastsam_single_point_inference( #Use point prompt to get the mask
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

def select_from_sam_everything_points( #Segments everything and merge masks that is closest to the point prompt
    model,
    point_prompt,
    input_img=None,
    imgsz=1408,
    iou=0.9,
    conf=0.4,
    max_distance=10,
    device=None,
    retina=True,
):
    print("Running SAM everything")
    # Run inference
    results = model(
        input_img,
        device=device,
        retina_masks=retina,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
    )
    print('Processing prompts')
    # Process prompts
    prompt_process = FastSAMPrompt(input_img, results, device=device)
    
    ann = prompt_process.everything_prompt()
    ann = ann.cpu().numpy()
    #print('Input image should be 1408x1408: ', input_img.shape)
    print('ann image should be 1408x1408: ', ann.shape)
    
    image_size = input_img.shape[0] * input_img.shape[1]
    #print("image_size: ", image_size)
    filtered_centers = []  # Store centers instead of masks
    if ann is not None:
        for mask in ann:
            mask_indices = np.argwhere(mask)
            if len(mask_indices) > 0:
                within_radius = False
                for y, x in mask_indices:
                    if distance.euclidean(point_prompt, [x, y]) <= max_distance:
                        within_radius = True
                        break
                if within_radius:
                    mask_size = np.count_nonzero(mask)
                    if mask_size <= 0.2 * image_size:
                        # Calculate center of mass
                        center_y, center_x = np.mean(mask_indices, axis=0)
                        filtered_centers.append([center_x, center_y])  # Append center coordinates

    print('filtered centers: ', len(filtered_centers))
    return filtered_centers


def select_from_sam_everything(
    model,
    point_prompt,
    input_img=None,
    imgsz=1408,
    iou=0.9,
    conf=0.4,
    max_distance=10,
    max_mask_ratio = 0.2, # 20% of the image
    device=None,
    retina=True,
    include_largest_mask=False
):
    print("Running SAM everything")
    # Run inference
    results = model(
        input_img,
        device=device,
        retina_masks=retina,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
    )
    print('Processing prompts')
    # Process prompts
    prompt_process = FastSAMPrompt(input_img, results, device=device)
    
    ann = prompt_process.everything_prompt()
    try:
        ann = ann.cpu().numpy()
    except:
        print(ann)
        pass
    print('ann shape: ', ann.shape)
    
    image_size = input_img.shape[0] * input_img.shape[1]
    print("image_size: ", image_size)
    filtered_masks = []
    filtered_centers = []
    
    if ann is not None:
        # Find the largest mask if include_largest_mask is True
        if include_largest_mask:
            largest_mask = None
            largest_mask_size = 0
            for mask in ann:
                mask_size = np.count_nonzero(mask)
                if mask_size > largest_mask_size:
                    largest_mask_size = mask_size
                    largest_mask = mask
            if largest_mask is not None:
                filtered_masks.append(largest_mask)
                mask_indices = np.argwhere(largest_mask)
                center_y, center_x = np.mean(mask_indices, axis=0)
                filtered_centers.append([center_x, center_y])
                print("Included largest mask")
        
        # Filter masks based on point prompt
        for mask in ann:
            mask_indices = np.argwhere(mask)
            if len(mask_indices) > 0:
                within_radius = False
                for y, x in mask_indices:
                    for point in point_prompt:
                        if distance.euclidean(point, [x, y]) <= max_distance:
                            within_radius = True
                            break
                if within_radius:
                    mask_size = np.count_nonzero(mask)
                    if mask_size <= max_mask_ratio * image_size:  # Check if mask is not larger than 20%
                        filtered_masks.append(mask)
                        center_y, center_x = np.mean(mask_indices, axis=0)
                        filtered_centers.append([center_x, center_y])
                        print("Appended mask")
    
    # Combine masks
    combined_mask = np.zeros((input_img.shape[0], input_img.shape[1]), dtype=np.uint8)
    print('Filtered masks: ', len(filtered_masks))
    if filtered_masks:
        for mask in filtered_masks:
            combined_mask = np.logical_or(combined_mask, mask)
        return combined_mask, filtered_centers
    else:
        print("ERROR! NO MASKS FOUND")
        return combined_mask, filtered_centers
