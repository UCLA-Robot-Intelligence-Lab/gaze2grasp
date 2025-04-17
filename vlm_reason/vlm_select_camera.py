# ## VLM Integrations (Indepth)
# The VLM should be called 3 times:
# 1. gaze_processing/gaze_intent_visualization.py - VLM function call best choice given the gaze points
# 2. gaze_processing.select_camera - VLM select the optimal camera view
# 3. visualizations.live_visualization - User (manually) selects gaze points and VLM takes those images and figures out the optimal pose (out of top 4 options) 
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import sys
import PIL.Image
import json

load_dotenv

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def gemini_select_camera(image_path_1, image_path_2):

    prompt = """
        You will be given two camera views. Choose the view that is the most clear and unobstructed for the given task. Output the number 1 or 2 only. 
    """

    img_1 = PIL.Image.open(image_path_1)
    img_2 = PIL.Image.open(image_path_2)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt, img_1, img_2],
    )

    print(response.text)

if __name__ == "__main__":
    gemini_select_camera("/home/u-ril/gaze2grasp/vlm_images/coffee2_yellow/pcd_combined/grasp_lines_all_c.png")