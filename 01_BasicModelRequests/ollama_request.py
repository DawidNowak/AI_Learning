# Go to https://ollama.com/download to download the OLLAMA
# Then run in terminal: ollama run llama3.2

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
    "stream": False
}

response = requests.post(OLLAMA_API, json=payload, headers=HEADERS)
print(response.json()['message']['content'])
