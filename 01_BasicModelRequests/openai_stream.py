from dotenv import load_dotenv
from openai import OpenAI

MODEL_GPT = 'gpt-4o-mini'

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

stream = openai.chat.completions.create(
    model=MODEL_GPT,
    messages=messages,
    stream=True
)

for message in stream:
    print(message.choices[0].delta.content or '', end='', flush=True)