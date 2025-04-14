import cv2
import numpy as np
import os

base_link_color = [[0.2, 0.2, 0.8], [0.2, 0.8, 0.2], [0.8, 0.2, 0.2],
                   [0.8, 0.8, 0.2], [0.8, 0.2, 0.8], [0.2, 0.8, 0.8]]

class PixelSelector:
    def __init__(self):
        self.clicks = []
        self.color_index = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_thickness = 2
        self.line_type = cv2.LINE_AA

    def load_image(self, img, recrop=False, crop_center=None, crop_width=640, crop_height=480):
        self.img_original = img.copy()  # Keep a copy for saving without clicks
        self.img_display = img.copy()  # Keep a copy for displaying/drawing on
        if recrop and crop_center is not None:
            cropped_img_tuple = self.crop_at_point(img, crop_center[0], crop_center[1], width=crop_width, height=crop_height)
            self.img_original = cropped_img_tuple[0]
            self.img_display = cropped_img_tuple[1]
        else:
            self.img_original = img.copy()
            self.img_display = img.copy()
        self.clicks = []
        self.color_index = 0

    def crop_at_point(self, img, x, y, width=640, height=480):
        h, w = img.shape[:2]
        ymin = max(0, y - height // 2)
        ymax = min(h, y + height // 2)
        xmin = max(0, x - width // 2)
        xmax = min(w, x + width // 2)
        return img[ymin:ymax].copy(), img[xmin:xmax].copy()

    def draw_clicks(self):
        self.img_display = self.img_original.copy()
        for i, click in enumerate(self.clicks):
            x, y = click
            color_float = base_link_color[i % len(base_link_color)]
            color_to_use = tuple(int(c * 255) for c in reversed(color_float))  # Reversed for BGR
            cv2.circle(self.img_display, (x, y), 5, color_to_use, -1)
            cv2.circle(self.img_original, (x, y), 5, color_to_use, -1)
            text = str(i + 1)
            text_size = cv2.getTextSize(text, self.font, self.font_scale, self.font_thickness)[0]
            text_x = x - text_size[0] // 2 - 12
            text_y = y + text_size[1] // 2 - 12
            cv2.putText(self.img_display, text, (text_x, text_y), self.font, self.font_scale, color_to_use,
                        self.font_thickness, self.line_type)
        cv2.imshow("pixel_selector", self.img_display)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.clicks.append([x, y])
            self.draw_clicks()

    def run(self, img, initial_clicks=None, recrop=False, crop_center=None, crop_width=640, crop_height=480, save_path=None):
        self.load_image(img, recrop, crop_center, crop_width, crop_height)

        if initial_clicks is not None:
            self.clicks = list(initial_clicks)
            self.draw_clicks()
            if save_path:
                filename_base = os.path.splitext(os.path.basename(save_path))[0]
                dirname = os.path.dirname(save_path)
                cv2.imwrite(os.path.join(dirname, f"{filename_base}_original.png"), self.img_original)
                cv2.imwrite(os.path.join(dirname, f"{filename_base}_clicks.png"), self.img_display)
            cv2.imshow("Final Image with Initial Clicks", self.img_display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return self.clicks, self.img_original, self.img_display
        else:
            cv2.namedWindow('pixel_selector')
            cv2.setMouseCallback('pixel_selector', self.mouse_callback)
            while True:
                cv2.imshow("pixel_selector", self.img_display)
                k = cv2.waitKey(20) & 0xFF
                if k == 27:
                    break
            cv2.destroyAllWindows()
            if save_path:
                filename_base = os.path.splitext(os.path.basename(save_path))[0]
                dirname = os.path.dirname(save_path)
                cv2.imwrite(os.path.join(dirname, f"{filename_base}_original.png"), self.img_original)
                cv2.imwrite(os.path.join(dirname, f"{filename_base}_clicks.png"), self.img_display)
            return self.clicks, self.img_original, self.img_display

if __name__ == '__main__':
    # Create a dummy image for testing
    dummy_img = np.zeros((600, 800, 3), dtype=np.uint8)
    dummy_img[:] = (100, 100, 100)  # Gray background

    # Create a dummy save directory
    save_dir = "/home/u-ril/gaze2grasp/vlm_images/gaze_inputs"
    os.makedirs(save_dir, exist_ok=True)
    save_file_direct = os.path.join(save_dir, "test_image_direct.png")
    save_file_click = os.path.join(save_dir, "test_image_click.png")

    # Directly input an array of coordinates
    initial_coordinates = [[150, 100], [300, 250]]
    selector_direct = PixelSelector()
    clicked_pixels_direct, original_img_direct, clicks_img_direct = selector_direct.run(
        dummy_img, initial_clicks=initial_coordinates, save_path=save_file_direct
    )
    print("Direct input clicks:", clicked_pixels_direct)

    # Use the pixel selector for clicking
    selector_click = PixelSelector()
    clicked_pixels_click, original_img_click, clicks_img_click = selector_click.run(
        dummy_img, save_path=save_file_click
    )
    print("Clicked pixels:", clicked_pixels_click)