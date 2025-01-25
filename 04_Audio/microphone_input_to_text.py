from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import openai
import numpy as np
import tempfile
import soundfile as sf

load_dotenv()

openai = OpenAI()

def transcribe_audio(audio):
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

audio_input = gr.Audio(sources='microphone', type='numpy')
interface = gr.Interface(fn=transcribe_audio, inputs=audio_input, outputs='text', title='Voice to Chatbot', allow_flagging='never')
interface.launch()