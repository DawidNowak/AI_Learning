import os
import gradio as gr
from dotenv import load_dotenv
from IPython.display import Markdown, display, update_display
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig

load_dotenv()

output_dir = "./Outputs"
os.makedirs(output_dir, exist_ok=True)

AUDIO_MODEL = "whisper-1"
LLAMA = "meta-llama/Llama-3.2-3B-Instruct"

system_message = "You are an assistant that produces minutes of meetings from transcripts, with summary, key discussion points, takeaways and action items with owners, in markdown."

openai = OpenAI()

tokenizer = AutoTokenizer.from_pretrained(LLAMA)
tokenizer.pad_token = tokenizer.eos_token

def transcribe_audio(audio_file):
    with open(audio_file, "rb") as f:
        transcription = openai.audio.transcriptions.create(
            model=AUDIO_MODEL,
            file=f
        )
    return summarize(transcription.text)
    
def summarize(transcription):
    user_prompt = f"Below is an extract transcript of a Denver council meeting. Please write minutes in markdown, including a summary with attendees, location and date; discussion points; takeaways; and action items with owners.\n{transcription}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model = AutoModelForCausalLM.from_pretrained(LLAMA, device_map="auto")
    outputs = model.generate(inputs, max_new_tokens=2000)
    response = tokenizer.decode(outputs[0])

    return response


with gr.Blocks() as demo:
    gr.Markdown("### Meeting Minutes")
    audio_input = gr.Audio(label="Upload Audio", type="filepath")

    output = gr.TextArea("Output...", label="Meeting Minutes", lines=30)
    
    def upload_callback(audio_file):
        return transcribe_audio(audio_file)
    
    audio_input.change(
        fn=upload_callback,
        inputs=[audio_input],
        outputs=[output])

demo.launch()