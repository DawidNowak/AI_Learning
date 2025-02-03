from dotenv import load_dotenv
from openai import OpenAI
import json
import gradio as gr

MODEL = "gpt-4o-mini"

load_dotenv()

openai = OpenAI()

default_topic = "cars"
default_number_of_data = 5
default_multishot_examples = [
    """{ "make": "Kia", "model": "Ceed", "year": 2018, "mileage": 50000 }""",
    """{ "make": "Honda", "model": "Civic", "year": 2020, "mileage": 30000 }""",
    """{ "make": "Ford", "model": "Mondeo", "year": 2022, "mileage": 40000 }"""
]

def is_valid_json(string):
    try:
        json.loads(string)
        return True
    except json.JSONDecodeError:
        return False

def generate_dataset(topic, number_of_data, ex1, ex2, ex3):
    examples = []
    invalid_inputs = []

    for i, ex in enumerate([ex1, ex2, ex3], start=1):
        if is_valid_json(ex):
            examples.append(json.loads(ex))
        elif ex not in ("", None):
            invalid_inputs.append(f"Example {i} is invalid.")

    if len(invalid_inputs) > 0:
        yield "\n".join(invalid_inputs)
        return

    system_prompt = f"""
    You are a helpful assistant whose main purpose is to generate datasets.
    Topic: {topic}
    Return the dataset strictly in JSON format.
    Include the following examples: '{examples}'
    and use only provided properties in generated objects.
    Return {number_of_data} examples each time.
    Do not repeat the provided examples.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Please generate my dataset for {topic}"}
    ]

    stream = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True)

    response_text = ""
    for message in stream:
        content = message.choices[0].delta.content or ''
        response_text += content
        yield response_text

gr_interface = gr.Interface(
    fn=generate_dataset,
    inputs=[
        gr.Textbox(label="Topic", value=default_topic, lines=2),
        gr.Number(label="Number of Examples", value=default_number_of_data, precision=0),
        gr.Textbox(label="Example 1", value=default_multishot_examples[0]),
        gr.Textbox(label="Example 2", value=default_multishot_examples[1]),
        gr.Textbox(label="Example 3", value=default_multishot_examples[2]),
    ],
    outputs=gr.Textbox(label="Generated Dataset", lines=20),
    flagging_mode='never'
)

gr_interface.launch()