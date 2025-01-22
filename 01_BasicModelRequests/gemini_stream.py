import os
from dotenv import load_dotenv
from openai import OpenAI

MODEL_GPT = "gemini-1.5-flash"

# Create an Google api key in Google AI Studio and store it in a .env file
# GOOGLE_API_KEY=AIza1234567890
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

system_prompt = "You are a very polite, courteous chatbot. You are here to help answer any questions even phylosophical ones."
question = "What is the meaning of life?"

geminiai = OpenAI(
    api_key=google_api_key, 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

messages = [
    {'role': 'system', 'content': system_prompt},
    {'role': 'user', 'content': question}
]

stream = geminiai.chat.completions.create(
    model=MODEL_GPT,
    messages=messages,
    stream=True
)

for message in stream:
    print(message.choices[0].delta.content or '', end='', flush=True)