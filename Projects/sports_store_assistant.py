import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import openai
import numpy as np
import tempfile
import soundfile as sf

load_dotenv()

openai = OpenAI()
MODEL = "gpt-4o-mini"

system_message = """
You are a helpful assistant at a sports equipment store called SportAI.
Give short, courteous answers, no more than 1 sentence.
Ask user if he needs any help or if you can help.
Always be accurate. If you don't know the answer, say so.
"""

def audio_to_text(audio):
    sample_rate, audio_array = audio  # Unpack the audio tuple
    
    # Check if audio is stereo (2D) and convert to mono if necessary
    if audio_array.ndim == 2:  # Check if audio has 2 dimensions (stereo)
        audio_array = np.mean(audio_array, axis=1)  # Convert to mono by averaging channels

    # Create a temporary file to save the audio data
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:

        # Save the audio array to the temporary file
        sf.write(temp_file.name, audio_array, sample_rate)
        temp_file.flush()

        with open(temp_file.name, 'rb') as audio_file:
            transcription = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language='pl'
            )
    return transcription.text

stock = {
    "bike": {
        "amateur": {"model": "Basic Bike", "price": "$299", "quantity": 3},
        "professional": {"model": "Professional Bike", "price": "$599", "quantity": 3},
        "elite": {"model": "Elite Bike", "price": "$1299", "quantity": 1}
    },
    "treadmill": {
        "amateur": {"model": "Home Fitness Treadmill", "price": "$399", "quantity": 0},
        "professional": {"model": "Gym-Quality Treadmill", "price": "$799", "quantity": 2},
        "elite": {"model": "High-Performance Treadmill", "price": "$1499", "quantity": 2}
    }
}

def get_stock_info(type, tier):
    print(f"type: {type}, tier: {tier}")
    equipment = stock.get(type.lower())
    if equipment:
        return equipment.get(tier.lower(), equipment)
    else:
        return "Type not found."
    
stock_info_tool = {
    "name": f"{get_stock_info.__name__}",
    "description": """Get stock information for a specific type of sports equipment 
        and tier level. Call this whenever a customer inquires about the availability, 
        model, price, or quantity of a specific category of equipment, for example, 
        when a customer asks 'What is the price and quantity of professional bikes?'.
        If the returned quantity is 0 it means this inventory is out of stock,
        inform user about that.""",
    "parameters": {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "description": "The type of equipment, e.g., 'bike' or 'treadmill', use singular form."
            },
            "tier": {
                "type": "string",
                "description": "The tier level of equipment, e.g., 'amateur', 'professional', or 'elite'."
            }
        },
        "required": ["type", "tier"],
        "additionalProperties": False
    }
}


tools = [{"type": "function", "function": stock_info_tool}]

def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    type = arguments.get('type')
    tier = arguments.get('tier')
    info = get_stock_info(type, tier)
    return {
        "role": "tool",
        "content": json.dumps({"type": type, "tier": tier, "stock_info": info}),
        "tool_call_id": tool_call.id
    }

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type='messages')
    msg = gr.Textbox()
    audio = gr.Audio(sources='microphone', type='numpy')

    with gr.Row():
        submit_btn = gr.Button("Submit", variant="primary")
        clear_btn = gr.ClearButton([audio, chatbot], variant="secondary")

    def chat(message, audio, history):
        if len(history) == 0:
            history.append({"role": "system", "content": system_message})

        if audio:
            message = audio_to_text(audio)

        history.append({"role": "user", "content": message})

        response = openai.chat.completions.create(
            model=MODEL,
            messages=history,
            tools=tools,
            stream=False)
        
        if response.choices[0].finish_reason == 'tool_calls':
            message = response.choices[0].message
            messages = list(history)
            messages.append({"role": "assistant", "content": "I'm checking inventory...", "tool_calls": message.tool_calls})
            tool_result = handle_tool_call(message)
            messages.append(tool_result)
            response = openai.chat.completions.create(
                model=MODEL,
                messages=messages,
                stream=False
            )
            
        history.append({"role": "assistant", "content": response.choices[0].message.content})

        return None, None, history
    
    msg.submit(chat, [msg, audio, chatbot], [msg, audio, chatbot])
    submit_btn.click(chat, [msg, audio, chatbot], [msg, audio, chatbot])

demo.launch()