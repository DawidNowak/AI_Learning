import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import openai
import numpy as np
import tempfile
import soundfile as sf
from io import BytesIO
from pydub import AudioSegment
from playsound import playsound
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

openai = OpenAI()

system_message = """Jesteś pomocnym asystentem, masz na imię Nova, Proszę odpowiadaj na pytania użytkownika i bądź zwięzła i prezycyjna.\
    Jeśli nie znasz odpowiedzi na pytanie, to o tym powiedz. Odpowiadaj wyłącznie w języku polskim."""

executor = ThreadPoolExecutor(max_workers=1)

def audio_to_text(audio):
    sample_rate, audio_array = audio  # Unpack the audio tuple
    
    # Check if audio is stereo (2D) and convert to mono if necessary
    if audio_array.ndim == 2:  # Check if audio has 2 dimensions (stereo)
        audio_array = np.mean(audio_array, axis=1)  # Convert to mono by averaging channels

    # Create a temporary file to save the audio data
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:

        # Save the audio array to the temporary file
        sf.write(temp_file.name, audio_array, sample_rate)
        temp_file.flush()

        with open(temp_file.name, 'rb') as audio_file:
            transcription = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language='pl'
            )
    return transcription.text

def text_to_audio(message):
    response = openai.audio.speech.create(
      model="tts-1",
      voice = "nova",
      input=message
    )
    
    audio_stream = BytesIO(response.content)
    audio = AudioSegment.from_file(audio_stream, format="mp3")

    # Save the file to a location and play it
    temp_file = "C:/AI/AI_Learning/04_Audio/Outputs/audio_chat_temp.wav"
    audio.export(temp_file, format="wav")
    playsound(temp_file)

    if os.path.exists(temp_file):
        os.remove(temp_file)


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type='messages')
    msg = gr.Textbox()
    audio = gr.Audio(sources='microphone', type='numpy')

    with gr.Row():
        submit_btn = gr.Button("Submit", variant="primary")
        clear_btn = gr.ClearButton([msg, audio, chatbot], variant="secondary")

    def chat(message, audio, history):
        if len(history) == 0:
            history.append({"role": "system", "content": system_message})

        if audio:
            message = audio_to_text(audio)

        history.append({"role": "user", "content": message})

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=history,
            stream=False)

        result = response.choices[0].message.content
        history.append({"role": "assistant", "content": result})

        executor.submit(text_to_audio, result)

        return "", history
    
    msg.submit(chat, [msg, audio, chatbot], [msg, chatbot])
    submit_btn.click(chat, [msg, audio, chatbot], [msg, chatbot])

demo.launch()