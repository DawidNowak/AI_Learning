from dotenv import load_dotenv
from openai import OpenAI

MODEL_GPT = 'gpt-4o-mini'

# Create an OpenAI api key and store it in a .env file
# OPENAI_API_KEY=sk-proj-1234567890
load_dotenv(override=True)

system_prompt = 'You are the expert at python in coding. You are asked to explain the following code to a beginner in a simple way.'

question = """
Please explain what this code does and why:
yield from {book.get("author") for book in books if book.get("author")}
"""

messages = [
    {'role': 'system', 'content': system_prompt},
    {'role': 'user', 'content': question}
]

openai = OpenAI()

response = openai.chat.completions.create(
    model=MODEL_GPT,
    messages=messages,
    stream=False
)

print(response.choices[0].message.content)