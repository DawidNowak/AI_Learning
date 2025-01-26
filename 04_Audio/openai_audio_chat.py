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

load_dotenv()

openai = OpenAI()

system_message = """You are a helpfull assistant, your name is Bob, please respond to questions, be precise and concise. If you don't know the answer, just say so."""

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
                file=audio_file
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

def chat(message, history, audio):

    print("\n******history")
    if history is not None:
        for entry in history:
            print(f"entry: {entry}")
    else:
        history = []
        print('Empty history')

    if audio:
        transcription = audio_to_text(audio)
        message = transcription
    print("******")
    print(f"message: {message}")

    messages = [{"role": "system", "content": system_message}]
    if history is not None:
        messages.__add__(history)
    messages.append({"role": "user", "content": message})

    history = messages

    print("******messages")
    for msg in messages:
        print(f"{msg}")
        
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=False)

    return response, history

chat_interface = gr.Interface(
    fn=chat,
    inputs=[
        gr.Textbox(label="Message"),
        gr.State(),  # To hold the chat history
        gr.Audio(sources='microphone', type='numpy')
    ],
    outputs=[
        gr.Chatbot(label="Chat"),
        gr.State()  # To hold the updated chat history
    ],
    title="Chatbot",
    description="This is a voice-enabled chatbot."
)

chat_interface.launch()
