from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

openai = OpenAI()

audio_file= open("./Outputs/motivational_quote.wav", "rb")

transcription = openai.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file
)

print(transcription)

# print(transcription.text)