import os
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

MODEL_GPT = "deepseek-reasoner"

system_message = """Jesteś asystentem twórcy materiałów na youtube.
Będę podawał Ci transkrypcję z moich filmów,
a Twoim zadaniem jest podsumowanie filmu oraz
wypisanie kluczowych informacji wraz z informacją,
w której minucie/sekundzie filmu te informacje się znajdują.
Odpowiadaj w formacie Markdown.
"""

def get_summarization(id):
    transcript = YouTubeTranscriptApi.get_transcript(id, languages=['pl'])

    deepseekai = OpenAI(
        api_key=deepseek_api_key, 
        base_url="https://api.deepseek.com"
    )

    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': f"{transcript}"}
    ]

    stream = deepseekai.chat.completions.create(
        model=MODEL_GPT,
        messages=messages,
        stream=True
    )

    response_text = ''
    for message in stream:
        content = message.choices[0].delta.content or ''
        response_text += content
        yield response_text

with gr.Blocks() as demo:
    gr.Markdown("# Podsumowanie filmu YouTube")
    
    with gr.Row():
        youtube_id = gr.Textbox(
            label="YouTube Video ID",
            placeholder="Podaj ID (np. 9npbvaayuxU&ab)"
        )
        submit_btn = gr.Button("Potwierdź", variant="primary")
    
    markdown = gr.Markdown("## Podsumowanie\n*Twoje podsumowanie pojawi się tutaj...*", elem_id="output")

    submit_btn.click(
        fn=get_summarization,
        inputs=[youtube_id],
        outputs=[markdown]
    )
    youtube_id.submit(
        fn=get_summarization,
        inputs=[youtube_id],
        outputs=[markdown]
    )

demo.launch(share=True)