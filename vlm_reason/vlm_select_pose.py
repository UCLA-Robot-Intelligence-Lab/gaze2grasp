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

def gemini_select_pose(image_path):

    prompt = """
        You will be given an image of an environment with a robot arm, along with multiple possible grasp poses predicted by an external pose model. Based on the environment, identify the optimal pose. You will identify each potential pose by the distinguishable color.

        Stop for a second. Look at the image, and describe in extreme detail every single thing you see along with every single colored ball you see on screen, and their locations relative to the robot arm, to other objects, and to other colored balls

        Here's the thought process for how you should identify the optimal pose.

        1. What is the object you are trying to grasp? Use the context of the environment to figure out and reinforce your assumption of what it is. is it fragile? is there anything about its shape that would make certain poses difficult?
        2. For each pose estimated, would this pose work or would it not? Is there anything about the object's shape that would make this specific pose un-ideal?
        3. Look beyond the poses. What are the other objects in the environment? Will a certain grasp pose be beneficial for future (potential) interactions with other items? For example, if it's most likely a pick and place task, ensure that the optimal pick pose is also the optimal place pose for wherever the object might be placed. Assume that the pick pose is the same as the place pose. Analyze approach angles for placement locations. Factor in environment obstructions or physical object limitations (of the potential target place location) in your analysis of this pose. 

        For each grasp pose you see in the image, you will reason on whether or not it is a good pose, and how it compares to other poses in the same image. You MUST only return 1 single pose.

        The above instructions must be considered silently. Your output will be parsed as a json by our postprocessing functions, so absolutely 100% sure that your response follows the below JSON schema. Do NOT output anything besides your final JSON schema response.

        # Use this JSON schema:
        **One example:**
        {
            "Detected Poses": "red, blue, green",
            "Individual Reasoning: {
                {"id": "red", "analysis": "the red pose indicates the robot is attempting to grasp the item from the top. However, this is problematic because the object is cylindrical with an outward taper. It would make more sense to grasp this from the side"},
                {"id": "blue", "analysis": "this grasp is valid because it approaches from the side"},
                {"id": "green", "analysis": "this grasp is also valid, but likely less optimal than blue because it attempts to grasp the cylinder from an angle, which may cause issues later on when the robot releases this object"}
            },
            "Final Pose": "blue", "justification": "the blue pose makes the most sense, and is the most logical approach with future possibilities considered."
        }

        **Another example:**
        {
            "Detected Poses": "pink, aqua, orange",
            "Individual Reasoning: {
                {"id": "pink", "analysis": "this pose indicates the robot is attempting to grasp the item from the side. However, this is problematic because there appears to Be containers in the background. This side pose would make it difficult to drop the object into the containers."},
                {"id": "aqua", "analysis": "this grasp is valid because it approaches from the top"},
                {"id": "orange", "analysis": "this grasp is also valid, because it approaches from the top."}
            },
            "Final Pose": "pink", "justification": "both the aqua and orange poses are valid, but let's go with pink"
        }
    """

    img = PIL.Image.open(image_path)

    response = client.models.generate_content(
        model="gemini-2.0-flash", # has to be changed to gemini-2.5-pro later
        contents=[prompt, img],
    )

    parse = response.text[7:-3]
    try:
        json_response = json.loads(response.text[7:-3])
        #print(json_response)
        print(f'Final guess: {json_response["Final Pose"]}')
        print(f'Justification: {json_response["justification"]}')
        return json_response["Final Pose"]
    except Exception as e:
        print('Something went wrong')
        print(e)
        return None

if __name__ == "__main__":
    gemini_select_pose("/home/u-ril/gaze2grasp/vlm_images/coffee2_yellow/pcd_combined/grasp_lines_all_c.png")