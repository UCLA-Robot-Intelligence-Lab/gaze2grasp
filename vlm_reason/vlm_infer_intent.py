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

load_dotenv

# =======================================================================
# CREATE FUNCTION DECLARATIONS FOR MODEL
# =======================================================================

# import all of the api functions
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# make sure to do full imports, because each file has to do some set up outside of their function (maybe change this later?)
from api import open_micro_contact_graspnet
from api import open_microwave
from api import pick_micro_contact_graspnet
from api import pouring

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

open_micro_contact_graspnet_declaration = {
    "name": "open_micro_contact_graspnet",
    "description": "Opens micro contact graspnet", # fix this description later
}

open_microwave_declaration = {
    "name": "open_microwave",
    "description": "Opens micro contact graspnet", # fix this description later
}

pick_micro_contact_graspnet_declaration = {
    "name": "pick_micro_contact_graspnet",
    "description": "Pick micro contact graspnet", # fix this description later
}

pouring_declaration = {
    "name": "pouring",
    "description": "perform pouring action", # fix this description later
}

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

tools = types.Tool(function_declarations=[open_micro_contact_graspnet_declaration, 
                                          open_microwave_declaration, 
                                          pick_micro_contact_graspnet_declaration, 
                                          pouring_declaration])
config = types.GenerateContentConfig(tools=[tools])

response = client.models.generate_content(
     model="gemini-2.0-flash",
     contents="Figure out how long the name 'Hoangzhou' is",
     config=config,
 )

# print(response.candidates[0].content_parts[0].function_call)
if response.candidates[0].content.parts[0].function_call:
    function_call = response.candidates[0].content.parts[0].function_call
    print(f"Function to call: {function_call.name}")
    print(f"Arguments: {function_call.args}")

    if function_call.name == "open_micro_contact_graspnet":
        open_micro_contact_graspnet.open_micro_contact_graspnet(**function_call.args)
    elif function_call.name == "open_microwave":
        open_microwave.open_microwave(**function_call.args)
    elif function_call.name == "pick_micro_contact_graspnet":
        pick_micro_contact_graspnet.pick_micro_contact_graspnet(**function_call.args)
    elif function_call.name == "pouring":
        pouring.pour(**function_call.args)
    
else:
    print("No function call found in the response.")
    print(response.text)