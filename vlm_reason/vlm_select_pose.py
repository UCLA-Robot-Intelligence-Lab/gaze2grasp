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

from groq import Groq
import base64

load_dotenv()

def gemini_select_pose(image_path):

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    prompt = """
        **VLM Task:** Analyze the provided image, which shows a robot arm, a target object, and several potential grasp poses indicated by colored markers. Your goal is to identify the single optimal grasp pose for the inferred task, considering the entire environment and the object's properties.

        **Internal Reasoning Process (Follow these steps silently, DO NOT include them in the output):**

        1.  **Scene Description:** Silently, meticulously observe the image. Identify the robot arm, the target object, all colored grasp markers, and *all other objects and significant environmental features* (e.g., containers, fixtures, surfaces, walls, potential obstructions). Note their spatial relationships.
        2.  **Task Inference & Object Analysis:**
            *   What is the most likely target object based on its appearance and context?
            *   What is the likely task given the object and the surrounding environment? (e.g., Is it likely a pick-and-place? Insertion? Handover? Stacking?). Use environmental cues (like containers, specific machinery, designated drop-off zones) to infer the task and the likely destination or interaction point.
            *   What are the object's relevant properties? (Fragile? Heavy? Specific shape constraints? Does its shape suggest a specific orientation for placement or use?).
        3.  **Environmental Constraints & Placement/Interaction Analysis:**
            *   Identify the likely placement location or interaction area based on the inferred task and environmental cues.
            *   Analyze this target location/area: Is it constrained? Is there a required entry angle or orientation for placing/inserting the object? Are there obstructions *near this target area* that the robot arm, gripper, or the *object being held* might collide with during the final placement/interaction motion? (e.g., narrow opening, overhead structure, surrounding items).
            *   Consider the approach path for both picking *and* placing/interacting. Are there obstacles the arm needs to avoid during transit or the final approach?
        4.  **Pose Evaluation (for each colored pose):**
            *   **Grasp Stability:** Is this a stable grasp on the object itself? Does it respect the object's shape, material, and potential fragility?
            *   **Pick Feasibility:** Can the robot arm physically reach this pose without colliding with the environment *during the approach to pick*?
            *   **Placement/Interaction Feasibility (CRITICAL):** Assuming this *same pose* is used for the final placement or interaction, can the robot successfully complete the inferred task? Consider the required entry angle/orientation for the target location, potential collisions with the environment *at the target location* (e.g., hitting the sides of a container, an overhead constraint), and clearance for the gripper and arm during the placing/interaction motion. Explicitly state *why* the environment makes this pose suitable or unsuitable for the final step of the task.
            *   **Comparison:** How does this pose compare to the others regarding stability, pick clearance, and *especially* placement/interaction feasibility within the environmental context?
        5.  **Final Selection:** Choose the single pose that is most stable, collision-free for the entire task (pick, transit, place/interact), and best suited for the inferred task's final requirements based on the environmental analysis.

        **Output Requirements:**

        *   Your response MUST be a single, valid JSON object.
        *   Do NOT output any text before or after the JSON object.
        *   Strictly adhere to the JSON schema provided below.
        *   The `analysis` for each pose *must* include reasoning about environmental factors and the feasibility of the *entire* inferred task (especially placement or final interaction).
        *   The final `justification` *must* explicitly explain why the chosen pose is optimal in the context of the environment and the inferred task, highlighting how it addresses potential environmental constraints or placement requirements better than rejected poses.

        **JSON Schema to Use:**

        ```json
        {
        "detected_poses": [
            "color1",
            "color2",
            "color3" 
        ],
        "pose_analysis": [
            {
            "id": "color1",
            "analysis": "Reasoning for this pose. Example: The red pose is a top-down grasp on the object. While potentially stable for picking, the likely placement target appears to be a narrow container opening directly below an overhead structural element in the environment. Approaching to place with this top-down grasp would likely cause the robot's gripper or arm to collide with this overhead element. Therefore, this pose is unsuitable for the placement part of the inferred task."
            },
            {
            "id": "color2",
            "analysis": "Reasoning for this pose. Example: The blue pose is a side grasp. This approach seems to avoid collision with the overhead element identified near the placement container. The grasp appears stable on the object's side. This allows for insertion into the container from the side/front, which seems necessary given the environmental constraints. This pose appears feasible for both pick and place."
            },
            {
            "id": "color3",
            "analysis": "Reasoning for this pose. Example: The green pose is an angled grasp from the front. While potentially avoiding the overhead collision, this grasp orientation might make precise alignment with the narrow container opening difficult during placement, and stability could be lower than a direct side grasp depending on the object's shape. It is less optimal than the blue pose due to potential placement inaccuracy or instability."
            }
        ],
        "final_selected_pose": "chosen_color",
        "justification": "Summary justification. Example: The blue pose is selected as optimal. It provides a stable side grasp on the object and, crucially, allows for placement into the inferred target container without colliding with the overhead environmental structure identified as a key constraint. Top-down grasps (like red) are infeasible due to this collision risk during placement. Angled grasps (like green) are less desirable due to potential instability or difficulty aligning with the placement target compared to the direct side grasp facilitated by blue."
        }
        ```
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
    
def llama_select_pose(image_path):

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    prompt = """
        **VLM Task:** Analyze the provided image, which shows a robot arm, a target object, and several potential grasp poses indicated by colored markers. Your goal is to identify the single optimal grasp pose for the inferred task, considering the entire environment and the object's properties.

        **Internal Reasoning Process (Follow these steps silently, DO NOT include them in the output):**

        1.  **Scene Description:** Silently, meticulously observe the image. Identify the robot arm, the target object, all colored grasp markers, and *all other objects and significant environmental features* (e.g., containers, fixtures, surfaces, walls, potential obstructions). Note their spatial relationships.
        2.  **Task Inference & Object Analysis:**
            *   What is the most likely target object based on its appearance and context?
            *   What is the likely task given the object and the surrounding environment? (e.g., Is it likely a pick-and-place? Insertion? Handover? Stacking?). Use environmental cues (like containers, specific machinery, designated drop-off zones) to infer the task and the likely destination or interaction point.
            *   What are the object's relevant properties? (Fragile? Heavy? Specific shape constraints? Does its shape suggest a specific orientation for placement or use?).
        3.  **Environmental Constraints & Placement/Interaction Analysis:**
            *   Identify the likely placement location or interaction area based on the inferred task and environmental cues.
            *   Analyze this target location/area: Is it constrained? Is there a required entry angle or orientation for placing/inserting the object? Are there obstructions *near this target area* that the robot arm, gripper, or the *object being held* might collide with during the final placement/interaction motion? (e.g., narrow opening, overhead structure, surrounding items).
            *   Consider the approach path for both picking *and* placing/interacting. Are there obstacles the arm needs to avoid during transit or the final approach?
        4.  **Pose Evaluation (for each colored pose):**
            *   **Grasp Stability:** Is this a stable grasp on the object itself? Does it respect the object's shape, material, and potential fragility?
            *   **Pick Feasibility:** Can the robot arm physically reach this pose without colliding with the environment *during the approach to pick*?
            *   **Placement/Interaction Feasibility (CRITICAL):** Assuming this *same pose* is used for the final placement or interaction, can the robot successfully complete the inferred task? Consider the required entry angle/orientation for the target location, potential collisions with the environment *at the target location* (e.g., hitting the sides of a container, an overhead constraint), and clearance for the gripper and arm during the placing/interaction motion. Explicitly state *why* the environment makes this pose suitable or unsuitable for the final step of the task.
            *   **Comparison:** How does this pose compare to the others regarding stability, pick clearance, and *especially* placement/interaction feasibility within the environmental context?
        5.  **Final Selection:** Choose the single pose that is most stable, collision-free for the entire task (pick, transit, place/interact), and best suited for the inferred task's final requirements based on the environmental analysis.

        **Output Requirements:**

        *   Your response MUST be a single, valid JSON object.
        *   Do NOT output any text before or after the JSON object.
        *   Strictly adhere to the JSON schema provided below.
        *   The `analysis` for each pose *must* include reasoning about environmental factors and the feasibility of the *entire* inferred task (especially placement or final interaction).
        *   The final `justification` *must* explicitly explain why the chosen pose is optimal in the context of the environment and the inferred task, highlighting how it addresses potential environmental constraints or placement requirements better than rejected poses.

        **JSON Schema to Use:**
        In this json format, do not output any additional spaces, tabs, or new lines just have one space after and before a comma, bracket, or squiggly bracket. REMEMBER after commas delete spaces and after squiggly brackets { and square brackets [ delete spaces.
        Delete all spaces that is not affecting a sentence. Between different variables such as detected poses, pose_analysis, final_selected_pose, and justification there should be no spaces. 
        
        ```json
        {"detected_poses": ["color1", "color2","color3"],"pose_analysis":[{"id": "color1", "analysis": "Reasoning for this pose. Example: The red pose is a top-down grasp on the object. While potentially stable for picking, the likely placement target appears to be a narrow container opening directly below an overhead structural element in the environment. Approaching to place with this top-down grasp would likely cause the robot's gripper or arm to collide with this overhead element. Therefore, this pose is unsuitable for the placement part of the inferred task."}, {"id": "color2", "analysis": "Reasoning for this pose. Example: The blue pose is a side grasp. This approach seems to avoid collision with the overhead element identified near the placement container. The grasp appears stable on the object's side. This allows for insertion into the container from the side/front, which seems necessary given the environmental constraints. This pose appears feasible for both pick and place."}, {"id": "color3", "analysis": "Reasoning for this pose. Example: The green pose is an angled grasp from the front. While potentially avoiding the overhead collision, this grasp orientation might make precise alignment with the narrow container opening difficult during placement, and stability could be lower than a direct side grasp depending on the object's shape. It is less optimal than the blue pose due to potential placement inaccuracy or instability."}], "final_selected_pose": "chosen_color", "justification": "Summary justification. Example: The blue pose is selected as optimal. It provides a stable side grasp on the object and, crucially, allows for placement into the inferred target container without colliding with the overhead environmental structure identified as a key constraint. Top-down grasps (like red) are infeasible due to this collision risk during placement. Angled grasps (like green) are less desirable due to potential instability or difficulty aligning with the placement target compared to the direct side grasp facilitated by blue."}
        ```
    """
    base64_image = encode_image(image_path)
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"),)
    
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        }
                    }
                ]
            },
        ],
        temperature=0.5,
        max_completion_tokens=512,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )

    completion = str(completion.choices[0].message).replace('\\n', '').strip()
    print(completion)
    completion = completion.split("""```',""")[0]

    print(completion)
    try:
        json_response = json.loads(completion)
        #print(json_response)
        print(f'Final guess: {json_response["Final Pose"]}')
        print(f'Justification: {json_response["justification"]}')
        return json_response["Final Pose"]
    except Exception as e:
        print('Something went wrong')
        print(e)
        return None

if __name__ == "__main__":
    #gemini_select_pose("/home/u-ril/gaze2grasp/vlm_images/coffee3_pink/pcd56/grasp_lines_all_.png")
    llama_select_pose("/home/u-ril/gaze2grasp/vlm_images/coffee3_pink/pcd56/grasp_lines_all_.png")