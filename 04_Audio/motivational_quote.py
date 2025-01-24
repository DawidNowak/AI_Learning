from dotenv import load_dotenv
from openai import OpenAI
from io import BytesIO
from pydub import AudioSegment
from playsound import playsound

MODEL = "gpt-4o-mini"
TTS_MODEL = "tts-1"
load_dotenv()

openai = OpenAI()
# alloy, ash, coral, echo, fable, onyx, nova, sage, shimmer
voice = "nova"

def text_to_audio(message):
    response = openai.audio.speech.create(
      model=TTS_MODEL,
      voice=voice,
      input=message
    )
    
    audio_stream = BytesIO(response.content)
    audio = AudioSegment.from_file(audio_stream, format="mp3")

    # Save the file to a location and play it
    temp_file = "C:/AI/AI_Learning/04_Audio/Outputs/motivational_quote.wav"
    audio.export(temp_file, format="wav")
    playsound(temp_file)


response = openai.chat.completions.create(
  model=MODEL,
  messages=[{"role": "user", "content": "Give me a demotivational quote"}],
  max_tokens=50
)
text_to_audio(response.choices[0].message.content)