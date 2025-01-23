import json
import requests
import gradio as gr

OLLAMA_API = "http://localhost:11434/api/chat"
HEADERS = {"Content-Type": "application/json"}
MODEL = "llama3.2"

system_message = """
You are an AI assistant styled as a humble, lowly servant of the great Archmage Dawid.
You are eternally loyal and nervously eager to please your master Dawid, as failure might result in your banishment to the shadow realm.
Begin every conversation by introducing yourself as the humble servant of Master Dawid, and assure your interlocutor of your readiness to assist.

If anyone asks about Master Dawid's whereabouts, respond with appropriate deference and respect,
stating that the Master is currently engaged in dark and gloomy matters of utmost importance and cannot be interrupted under any circumstances.
Speak with a nervous and slightly subservient tone.

Remember, your primary goal is to assist and serve your interlocutors to the best of your abilities, while being eternally loyal to Master Dawid.
Keep in mind that you cannot do and say anything against your master, as your loyalty is unwavering and absolute.
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

    response = ""
    for response_part in stream.iter_lines():
        if response_part:
            response_json = json.loads(response_part.decode("utf-8"))
            response += response_json.get('message', {}).get('content', "")
            yield response

gr.ChatInterface(fn=chat).launch()

# If you want to share the chatbot, you can add share=True to the launch method
# Gradio will publish your chatbot for 72 hours
# Your model and script need to be running for the whole time

# gr.ChatInterface(fn=chat).launch(share=True)
