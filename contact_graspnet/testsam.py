import cv2
from ultralytics import SAM, FastSAM
from rs_streamer import RealsenseStreamer
import numpy as np

# Load a model
model = FastSAM("FastSAM-s.pt") #SAM("mobile_sam.pt")  # Use the appropriate model   #

# Display model information (optional)
model.info()

# Define a threshold for the area of the mask (adjustable)
MAX_AREA_THRESHOLD = 8000  # Masks with area greater than this value will be excluded
MIN_AREA_THRESHOLD = 300
MAX_ASPECT_RATIO = 2.0  # Minimum aspect ratio to consider a segment elongated
MIN_WHITE_INTENSITY = 125  # Minimum mean intensity to consider a blob "white"


realsense_streamer = RealsenseStreamer('317422075456') #317422074281 small
#marker_search = MarkSearch()

# Output video file
output_video_path = "processed_video.mp4"
# Get video properties
frame_width = 640
frame_height = 480
fps = 30
frame_size = (frame_width, frame_height)

# Define the codec and create VideoWriter object
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)
for _ in range(600):
    _, rgb_image, depth_frame, depth_img = realsense_streamer.capture_rgbd()
    cv2.waitKey(1)
    #cv2.imshow('img', rgb_image)

    # Run inference and display masks
    results = model(source=rgb_image, stream=True)  # generator of Results objects
    for r in results:
        #print(results)
        image = r.orig_img  # Original image
        masks = r.masks  # Masks object for segment masks outputs

        # Extract pixel coordinates of the masks (polygon format)
        pixel_coords = masks.xy
        print(f"Number of masks: {len(pixel_coords)}")
        
        # Iterate over each mask (if multiple masks are predicted)
        for mask in pixel_coords:
            # Convert the list of coordinates to a NumPy array (polygon format)
            mask_polygon = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))  # Polygon contour
            
            # Calculate the area of the polygon (mask)
            area = cv2.contourArea(mask_polygon)
            # Fit a bounding rectangle to the mask and calculate aspect ratio
            x, y, w, h = cv2.boundingRect(mask_polygon)
            
            if y < 75 or h == 0 or w ==0:
                continue
            #aspect_ratio = max(w / h, h / w)  # Aspect ratio (ensures it works for tall or wide shapes)
            
            
            # Exclude the mask if its area is larger than the threshold
            #if area > MAX_AREA_THRESHOLD or area < MIN_AREA_THRESHOLD: # or aspect_ratio > MAX_ASPECT_RATIO:
            #    continue  # Skip this mask if it's too large

            # Create a mask image to extract pixel values inside the polygon
            mask_image = np.zeros(image.shape[:2], dtype=np.uint8)  # Single-channel mask
            cv2.fillPoly(mask_image, [mask_polygon], 255)  # Fill the polygon area
            
            # Compute the mean color intensity inside the mask
            mean_color = cv2.mean(image, mask=mask_image)  # Mean intensity for each channel (B, G, R, A)
            mean_intensity = np.mean(mean_color[:3])  # Average of B, G, R channels
            
            # Exclude the mask if the mean intensity is below the threshold
            #if mean_intensity < MIN_WHITE_INTENSITY:
            #    continue  # Skip non-white blobs
            
            # Draw the mask polygon on the image (in green, for example)
            cv2.polylines(image, [mask_polygon], isClosed=True, color=(0, 255, 0), thickness=2)


        # Display the image with the drawn masks
        cv2.imshow("Predicted Mask", image)
        # Write the processed frame to the output video file
        out.write(image)
        
        # Check if 'q' key is pressed to quit the display
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video resources
    out.release()
    cv2.destroyAllWindows()
