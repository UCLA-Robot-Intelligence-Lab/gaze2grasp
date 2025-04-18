# ## VLM Integrations (Indepth)
# The VLM should be called 3 times:
# 1. gaze_processing/gaze_intent_visualization.py - VLM function call best choice given the gaze points
# 2. gaze_processing.select_camera - VLM select the optimal camera view
# 3. visualizations.live_visualization - User (manually) selects gaze points and VLM takes those images and figures out the optimal pose (out of top 4 options) 
from google import genai
from google.genai import types
from openai import OpenAI

from dotenv import load_dotenv
import os
import sys
import base64
import PIL.Image

load_dotenv

# =======================================================================
# CREATE FUNCTION DECLARATIONS FOR MODEL
# =======================================================================

# import all of the api functions
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# NOTE: ACTUAL FUNCTION CALLING HAS BEEN TEMPORARILY DISABLED FOR BENCHMARKING, IF YOU WANT TO CALL THE ACTUAL FUNCTION UNCOMMNET THE BELOW LINES AND COMMENT THE REDEFINITIONS STARTING ON LINE 57
# make sure to do full imports, because each file has to do some set up outside of their function (maybe change this later?)
# from api import open_micro_contact_graspnet
# from api import open_microwave
# from api import pick_micro_contact_graspnet
# from api import pouring

"""
HERE'S HOW YOU DEFINE TOOL CALLING:
function_declaration = {
    "name": "function_name",
    "description": "function description",
    "parameters": {
        "type": "object",
        "properties": {
            "brightness": {
                "type": "integer",
                "description": "Light level from 0 to 100. Zero is off and 100 is full brightness",
            },
            "color_temp": {
                "type": "string",
                "enum": ["daylight", "cool", "warm"],
                "description": "Color temperature of the light fixture, which can be `daylight`, `cool` or `warm`.",
            },
        },
        "required": ["brightness", "color_temp"],
    },
}
"""

def open_micro_contact_graspnet():
    print("open_micro_contact_graspnet() called")

def open_microwave():
    print("open_microwave() called")

def pick_micro_contact_graspnet():
    print("pick_micro_contact_graspnet() called")

def pouring():
    print("pour() called")

# open_micro_contact_graspnet_declaration = {
#     "name": "open_micro_contact_graspnet",
#     "description": "Opens micro contact graspnet", # fix this description later
# }


PROMPT = "You want to open the microwave"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def gemini_infer_intent(image):

    open_microwave_declaration = {
        "name": "open_microwave",
        "description": "Open the microwave", # fix this description later
    }
    pick_micro_contact_graspnet_declaration = {
        "name": "pick_micro_contact_graspnet",
        "description": "Use contact graspnet algorithm to figure out the optimal position to grab the microwave handle to open/close", # fix this description later
    }

    pouring_declaration = {
        "name": "pouring",
        "description": "perform pouring action", # fix this description later
    }


    function_declarations=[ open_microwave_declaration, 
                        pick_micro_contact_graspnet_declaration, 
                        pouring_declaration]

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    tools = types.Tool(function_declarations=function_declarations)
    config = types.GenerateContentConfig(tools=[tools])
    image = PIL.Image.open(image)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[PROMPT, image],
        config=config,
    )

    # print(response.candidates[0].content_parts[0].function_call)
    if response.candidates[0].content.parts[0].function_call:
        function_call = response.candidates[0].content.parts[0].function_call
        print(f"Function to call: {function_call.name}")
        print(f"Arguments: {function_call.args}")

        try:
            if function_call.name == "open_micro_contact_graspnet":
                open_micro_contact_graspnet.open_micro_contact_graspnet(**function_call.args)
            elif function_call.name == "open_microwave":
                open_microwave.open_microwave(**function_call.args)
            elif function_call.name == "pick_micro_contact_graspnet":
                pick_micro_contact_graspnet.pick_micro_contact_graspnet(**function_call.args)
            elif function_call.name == "pouring":
                pouring.pour(**function_call.args)
        except:
            pass
            
    else:
        print("No function call found in the response.")
        print(response.text)

def gpt_infer_intent(image, model='gpt-4o'):
    # available models: gpt-4o, o4-mini, o3, gpt-4.1
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    open_microwave_declaration = {
        "type": "function",
        "name": "open_microwave",
        "description": "Open the microwave", # fix this description later
        "parameters": {
            "type": "object",
            "properties": {
                "dummy_property": {
                    "type": "null",
                }
            },
            "required": [],
        },
    }

    pick_micro_contact_graspnet_declaration = {
        "type": "function",
        "name": "pick_micro_contact_graspnet",
        "description": "Use contact graspnet algorithm to figure out the optimal position to grab the microwave handle to open/close", # fix this description later
        "parameters": {
            "type": "object",
            "properties": {
                "dummy_property": {
                    "type": "null",
                }
            },
            "required": [],
        },
    }

    pouring_declaration = {
        "type": "function",
        "name": "pouring",
        "description": "perform pouring action", # fix this description later
        "parameters": {
            "type": "object",
            "properties": {
                "dummy_property": {
                    "type": "null",
                }
            },
            "required": [],
        },
    }

    function_declarations=[ open_microwave_declaration, 
                            pick_micro_contact_graspnet_declaration, 
                            pouring_declaration]

    tools = function_declarations
    
    # Getting the Base64 string
    base64_image = encode_image(image)

    response = client.responses.create(
        model=model,
        input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": PROMPT},
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{base64_image}",
            },
        ],
    }],
        tools=tools
    )

    print(response.output)

#def llama_infer_intent(image, model='')

gemini_infer_intent("/home/u-ril/gaze2grasp/vlm_images/coffee3_pink/pcd56/grasp_lines_all_.png")