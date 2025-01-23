import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

MODEL = "gpt-4o-mini"

load_dotenv()

openai = OpenAI()

system_message = """
You are a helpful assistant. Answer questions and if you don't know the answer, say so.
"""

def get_datetime():
    return datetime.now()

# Price checker tool definition
datetime_tool = {
    "name": f"{get_datetime.__name__}",
    "description": """Get current date and time.
        Call this whenever you need to know the actual date time,
        for example when a customer asks 'What time is it now?'"""
}

tools = [{"type": "function", "function": datetime_tool}]

def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    datetime = get_datetime()
    response = {
        "role": "tool",
        "content": json.dumps(datetime, default=str),
        "tool_call_id": tool_call.id
    }
    return response

def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)

    # response.choices[0].message
    #
    # ChatCompletionMessage(
    #   content=None,
    #   refusal=None,
    #   role='assistant',
    #   audio=None,
    #   function_call=None,
    #   tool_calls=[
    #       ChatCompletionMessageToolCall(
    #           id='call_OD7eow25Jb2JfrU4TiIh4yMQ',
    #           function=Function(
    #               arguments='{}',
    #               name='get_datetime'
    #           ),
    #           type='function')])

    if response.choices[0].finish_reason=="tool_calls":
        message = response.choices[0].message
        messages.append(message)
        response = handle_tool_call(message)
        messages.append(response)
        response = openai.chat.completions.create(model=MODEL, messages=messages)

    return response.choices[0].message.content

gr.ChatInterface(fn=chat, type="messages").launch()