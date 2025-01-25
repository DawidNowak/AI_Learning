import os
from dotenv import load_dotenv
from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image


MODEL = "dall-e-3"
load_dotenv()
openai = OpenAI()

OUTPUT_PATH = "./Outputs/"

def image_gen():
    image_response = openai.images.generate(
            model=MODEL,
            prompt="A cartoon-style image of a bald programmer with a brown beard and blue eyes, sitting at a desk learning AI. He is using a laptop with a cup of coffee next to him. The laptop screen shows an AI robot in a playful and animated design. The setting is cozy, with warm lighting, a bookshelf in the background, and a modern, minimalistic workspace.",
            size="1024x1024",
            n=1,
            response_format="b64_json",
        )
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data))

    # Ensure the directory exists
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    image.save(os.path.join(OUTPUT_PATH, "ai_learning.png"))

image_gen()