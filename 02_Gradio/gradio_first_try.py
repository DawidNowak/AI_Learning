from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai
import gradio as gr

load_dotenv()

openai = OpenAI()
google.generativeai.configure()

system_message = "You are a helpful assistant. Please respond to the questions from users. If you don't know the answer, just say so."

def message_gpt(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
      ]
    completion = openai.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
    )
    return completion.choices[0].message.content

def message_google(prompt):
    gemini = google.generativeai.GenerativeModel(
        model_name='gemini-1.5-flash',
        system_instruction=system_message
    )
    response = gemini.generate_content(prompt)
    return response.text

def get_ai_response(prompt, ai):
    if ai == "Google (free)":
        return message_google(prompt)
    elif ai == "GPT (paid)":
        return message_gpt(prompt)

gr.Interface(fn=get_ai_response, inputs=[
    gr.Textbox(lines=6, label="Prompt"),
    gr.Dropdown(["Google (free)", "GPT (paid)"], label="LLM")
], outputs=gr.TextArea(lines=10, label='Response'),
allow_flagging='never').launch()