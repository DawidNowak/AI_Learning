from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

MODEL = "gpt-4o-mini"

load_dotenv()

openai = OpenAI()

system_message = """You are a helpfull assistant, please respond to questions, be precise and concise. If you don't know the answer, just say so."""


def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    stream = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True)

    response_text = ""
    for message in stream:
        content = message.choices[0].delta.content or ''
        response_text += content
        yield response_text

gr.ChatInterface(fn=chat, type="messages").launch()