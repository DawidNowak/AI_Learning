import os
from dotenv import load_dotenv
import gradio as gr
import anthropic

MODEL = "claude-3-5-sonnet-20241022"

load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

question = """
You are the expert at python in coding. You are asked to explain the following code to a beginner in a simple way.

Please explain what this code does and why:
yield from {book.get("author") for book in books if book.get("author")}
"""

messages = [
    # No system message https://docs.anthropic.com/en/api/messages#body-messages
    {'role': 'user', 'content': question}
]

claude = anthropic.Anthropic(api_key=anthropic_api_key)

with claude.messages.stream(
    model=MODEL,
    max_tokens=200,
    messages=messages
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)