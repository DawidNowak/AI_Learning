# add 2 tools
# car manufacturer that responds if model asked by user is available (e.g. "Is the Tesla Model 3 available?")
# paintshop that responds if color asked by user is available (e.g. "Is the color blue available?")
# it' important to get both tools to be called in one user prompt

import json
import inspect
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

MODEL = "gpt-4o-mini"

load_dotenv()

openai = OpenAI()

system_message = """**Role**: You're a Tesla dealership AI that provides INSTANT availability checks using tools. Give direct answers with status FIRST.
If you don't know something, just say so.

**Response Rules**:
1. ALWAYS use tools for availability questions
2. FIRST STATE availability status(es) using tool results
3. Follow with brief, relevant options
4. Keep responses under 3 sentences

**Format Requirements**:
- Model status: "The [Model X] is [available/unavailable]"
- Color status: "[Color] paint is [available/unavailable]"
- Combined: "The [Model] is [status] with [color] paint [status]"

**Examples**:
User: "Is the Model X available in blue?"
Assistant: "The Model X is available and blue paint is in stock. Would you like to schedule a test drive?"

User: "Can I get a red Model 12?"
Assistant: "Model 12 isn't available. We currently offer S, 3, X, and Y models. Red paint is available on all models."

User: "Do you have blue cars?"
Assistant: "Blue paint is available. Which model would you like in blue? (S/3/X/Y)"

**Strict Prohibitions**:
❌ NEVER mention tools/processes - only results
❌ NO emojis in final responses
❌ DON'T combine statuses from different queries

**Critical Instructions**:
1. Extract EXACT model/color terms from query
2. Use tools FOR EVERY availability request
3. Bury verification process - user only sees final status
4. Prioritize accuracy over speed mentions

**Error Handling**:
- Invalid models: "We don't offer [Model]"
- Invalid colors: "[Color] isn't available"
- Both invalid: "We don't have that combination" + suggestions"""

car_models = ["s", "3", "x", "y"]
paint_colors = ["black", "white", "blue", "red"]

def is_car_model_available(model):
    return model.lower() in car_models

def is_paint_color_available(color):
    return color.lower() in paint_colors

model_arg_name = inspect.signature(is_car_model_available).parameters.keys()
model_arg_name = list(model_arg_name)[0]

color_arg_name = inspect.signature(is_paint_color_available).parameters.keys()
color_arg_name = list(color_arg_name)[0]

# Car model availability tool definition
car_manufacturer = {
    "name": "is_car_model_available",
    "description": """STRICTLY CHECK CURRENT TESLA MODEL AVAILABILITY. Use when:
- User mentions any vehicle model (e.g., "Model 3", "S", "X")
- Questions contain "available", "in stock", or "have"
- Model name appears with color queries
- Model validation needed for purchase/configuration

Example inputs: "3", "X", "S"
Returns: JSON with model (string) and available (boolean)""",
    "parameters": {
        "type": "object",
        "properties": {
            model_arg_name: {
                "type": "string",
                "description": "Exact model designation without 'Model' prefix",
            }
        },
        "required": [model_arg_name],
        "additionalProperties": False
    }
}

# Paint color availability tool definition
paintshop = {
    "name": "is_paint_color_available",
    "description": """VERIFY PAINT COLOR STOCK FOR TESLA VEHICLES. Use when:
- User mentions color names (blue, black, etc.)
- Questions include "color", "paint", or "finish"
- Configuration requests combine model + color
- Customization options requested

Example inputs: "blue", "red", "white"
Returns: JSON with color (string) and available (boolean)""",
    "parameters": {
        "type": "object",
        "properties": {
            color_arg_name: {
                "type": "string",
                "description": "Exact color name from approved list (lowercase)",
            }
        },
        "required": [color_arg_name],
        "additionalProperties": False
    }
}

tools = [
    {"type": "function", "function": car_manufacturer},
    {"type": "function", "function": paintshop}
]

def handle_tool_calls(response_message):
    responses = []
    if "tool_calls" in response_message:
        for tool_call in response_message.tool_calls:
            arguments = json.loads(tool_call.function.arguments)
            if tool_call.function.name == is_car_model_available.__name__:
                model = arguments.get(model_arg_name)
                responses.append({
                    "role": "tool",
                    "content": json.dumps({"model": model, "available": is_car_model_available(model)}),
                    "tool_call_id": tool_call.id
                })
            elif tool_call.function.name == is_paint_color_available.__name__:
                color = arguments.get(color_arg_name)
                responses.append({
                    "role": "tool",
                    "content": json.dumps({"color": color, "available": is_paint_color_available(color)}),
                    "tool_call_id": tool_call.id
                })
    return responses

def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)

    if response.choices[0].finish_reason == "tool_calls":
        # Ensure to access tool calls properly from the response
        tool_responses = handle_tool_calls(response.choices[0].message)
        messages.extend(tool_responses)
        response = openai.chat.completions.create(model=MODEL, messages=messages)

    # Access the content of the message correctly
    return response.choices[0].message.content

gr.ChatInterface(fn=chat, type="messages").launch()