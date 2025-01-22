# Go to https://ollama.com/download to download the OLLAMA
# Then run in terminal: ollama run llama3.2

import json
import requests

OLLAMA_API = "http://localhost:11434/api/chat"
HEADERS = {"Content-Type": "application/json"}
MODEL = "llama3.2"

system_message = "You are a helpfull AI assistant. You are here to help the user with any questions they might have. If you don't know the answer, just say so."
user_message = "How to get from Times Square to Central Park?"

messages = [
    {'role': 'system', 'content': system_message},
    {'role': 'user', 'content': user_message}
]

payload = {
    "model": MODEL,
    "messages": messages,
    "stream": True
}

stream = requests.post(OLLAMA_API, json=payload, headers=HEADERS, stream=True)

for response_part in stream.iter_lines():
    if response_part:
        response_json = json.loads(response_part.decode("utf-8"))
        print(response_json['message']['content'], end="", flush=True)
