import os
from dotenv import load_dotenv
from openai import OpenAI

MODEL_GPT = "deepseek-reasoner"

# Create an DeepSeek api key at https://platform.deepseek.com/ and store it in a .env file
# DEEPSEEK_API_KEY=sk-1234567890
load_dotenv()
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

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
question = "Good day sir, could you tell me what is this place?"

deepseekai = OpenAI(
    api_key=deepseek_api_key, 
    base_url="https://api.deepseek.com"
)

messages = [
    {'role': 'system', 'content': system_message},
    {'role': 'user', 'content': question}
]

stream = deepseekai.chat.completions.create(
    model=MODEL_GPT,
    messages=messages,
    stream=True
)

for message in stream:
    print(message.choices[0].delta.content or '', end='', flush=True)