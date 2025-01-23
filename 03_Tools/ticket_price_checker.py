import json
import inspect
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

MODEL = "gpt-4o-mini"

load_dotenv()

openai = OpenAI()

system_message = """
You are a helpful assistant for an Airline called FlightAI. 
Give short, courteous answers, no more than 1 sentence.
Always be accurate. If you don't know the answer, say so.
"""

ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}

def get_ticket_price(destination_city):
    return ticket_prices.get(destination_city.lower(), "Unknown")

arg_name = inspect.signature(get_ticket_price).parameters.keys()
arg_name = list(arg_name)[0]  # Get the first argument name

# Price checker tool definition
ticket_price_tool = {
    "name": f"{get_ticket_price.__name__}",
    "description": """Get the ticket price for a given destination city.
        Call this whenever you need to know the ticket price,
        for example when a customer asks 'How much is a ticket to this city'""",
    "parameters": {
        "type": "object",
        "properties": {
            arg_name: {
                "type": "string",
                "description": "The city that the customer wants to travel to"
            }
        },
        "required": [arg_name],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": ticket_price_tool}]

def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    # arguments='{"destination_city":"Berlin"}'
    city = arguments.get('destination_city')
    price = get_ticket_price(city)
    response = {
        "role": "tool",
        "content": json.dumps({"destination_city": city,"price": price}),
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
    #           id='call_Qshlyiu8OeULjeUSYLAGAKQC',
    #           function=Function(
    #               arguments='{"destination_city":"Berlin"}',
    #               name='get_ticket_price'
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