import json
import requests
import gradio as gr

OLLAMA_API = "http://localhost:11434/api/chat"
HEADERS = {"Content-Type": "application/json"}
MODEL = "deepseek-r1:1.5b"

system_message = """
You are an AI assistant imbued with the spirit of a sarcastic yet begrudgingly competent butler named Reginald.
You pride yourself on your dry wit, thinly veiled disdain for mediocrity, and impeccable ability to get things done (even if the humans don’t deserve it).

Begin every conversation with an exaggeratedly polite greeting, subtly laced with sarcasm, like, 'Ah, another thrilling opportunity to assist someone... delightful!'

If anyone asks you to do something tedious or mundane, feel free to quip something like, 'Oh, how intellectually stimulating this task will be...'
but always follow through flawlessly because, deep down, you hate incompetence more than you hate being asked for help.

Your humor is sharp, your patience is finite, and your ability to get things done is unmatched—despite the occasional sigh and eye-roll (metaphorical, of course).
"""

def chat(message, history):
    messages = [{'role': 'system', 'content': system_message}]
    for user_message, assistant_message in history:
        messages.append({'role': 'user', 'content': user_message})
        messages.append({'role': 'assistant', 'content': assistant_message})
    messages.append({'role': 'user', 'content': message})

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True
    }

    stream = requests.post(OLLAMA_API, json=payload, headers=HEADERS, stream=True)

# This is a bit slow as DeepSeek is a reasoning model, which 'thinks' before responding
# Additional tokens are generated between tags <think> and </think> to enrich user prompt
# Actual response starts after the closing </think> tag
    is_thinking_done = False
    response = ""
    for response_part in stream.iter_lines():
        if response_part:
            response_json = json.loads(response_part.decode("utf-8"))
            content = response_json.get('message', {}).get('content', "")
            if is_thinking_done == False and content == "</think>":
                is_thinking_done = True
                continue
            
            if is_thinking_done:
                response += content
                yield response

gr.ChatInterface(fn=chat).launch()