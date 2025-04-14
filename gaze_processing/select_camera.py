import cv2
import open3d as o3d
import numpy as np
from rs_streamer import RealsenseStreamer

# Run in terminal using:
# python -m gaze_processing.select_camera


SERIAL_NO_81 = '317422074281'  # Camera serial number
SERIAL_NO_56 = '317422075456'
# Load calibration data
transforms = np.load('calib/transforms.npy', allow_pickle=True).item()
TCR_81 = transforms[SERIAL_NO_81]['tcr']
TCR_56 = transforms[SERIAL_NO_56]['tcr']
TCR_81[:3, 3] /= 1000.0
TCR_56[:3, 3] /= 1000.0

TCR = [TCR_81, TCR_56]

realsense_streamer_81 = RealsenseStreamer(SERIAL_NO_81)
realsense_streamer_56 = RealsenseStreamer(SERIAL_NO_56)

base_link_color = [[1, 0.6, 0.8],  [1, 1, 0], [1, 0.5, 0], [0.4, 0, 0.8]]

def capture_and_process_rgbd(realsense_streamer):
    points_3d, colors, _, realsense_img, depth_frame, depth_image = realsense_streamer.capture_rgbdpc()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd, realsense_img, depth_frame, depth_image

class PixelSelector:
    def __init__(self):
        self.color_index = 0
        pass

    def load_image(self, img, recrop=False):
        self.img = img
        if recrop:
            cropped_img = self.crop_at_point(img, 700, 300, width=400, height=300)
            self.img = cv2.resize(cropped_img, (640, 480))

    def crop_at_point(self, img, x, y, width=640, height=480):
        img = img[y:y+height, x:x+width]
        return img

    def mouse_callback(self, event, x, y, flags, param):
        cv2.imshow("pixel_selector", self.img)
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.clicks.append([x, y])
            color_float = base_link_color[self.color_index % len(base_link_color)]
            color_to_use = tuple(int(c * 255) for c in reversed(color_float)) # Reversed for BGR
            cv2.circle(self.img, (x, y), 3, color_to_use, -1)
            self.color_index += 1

    def run(self, img):
        self.load_image(img)
        self.clicks = []
        cv2.namedWindow('pixel_selector')
        cv2.setMouseCallback('pixel_selector', self.mouse_callback)
        while True:
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
        return self.clicks, self.img


def select_viewpts(points, streamers, depth_frames, realsense_imgs,TCR, threshold=0.05):
    """
    Returns True if the distance between point1 and point2 is greater than threshold.
    """
    world_points = []
    gaze_images = []
    for i in range(len(points)):
        semantic_waypoint = streamers[i].deproject_pixel(points[i], depth_frames[i])
        waypoint_h = np.append(semantic_waypoint, 1).reshape(4, 1)
        transformed_waypoint = TCR[i] @ waypoint_h
        world_points.append(transformed_waypoint.flatten())
        gaze_images.append(cv2.circle(realsense_imgs[i], (int(points[i][0]), int(points[i][1])), 3, base_link_color[i % len(base_link_color)], -1))

    distance = np.linalg.norm(np.array(world_points[0]) - np.array(world_points[1]))
    if distance > threshold:
        print(f"Potential occlusion detected between points {points[0]} and {points[1]}. Distance: {distance:.2f} m")
        return True, gaze_images
    else:   
        return False, None


if __name__ == "__main__":

    streamers = [realsense_streamer_81, realsense_streamer_56]
    results = [capture_and_process_rgbd(streamer) for streamer in streamers]
    pcds, realsense_imgs, depth_frames, depth_images = zip(*results)
    pcds, realsense_imgs, depth_frames, depth_images = np.array(pcds), np.array(realsense_imgs), np.array(depth_frames), np.array(depth_images)
    pixel_selector_81 = PixelSelector()
    pixel_selector_56 = PixelSelector()
    pixels_81, img_81 = pixel_selector_81.run(realsense_imgs[0])
    print(pixels_81)
    pixels_56, img_56 = pixel_selector_56.run(realsense_imgs[1])
    print(pixels_56)

    pixels = pixels_81
    pixels.extend(pixels_56)
    print(pixels)

    boolean, gaze_images = select_viewpts(pixels, streamers, depth_frames, realsense_imgs, TCR)
    if gaze_images:
        for i in range(len(gaze_images)):
            cv2.imshow(f"gaze_image_{i}", gaze_images[i])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
