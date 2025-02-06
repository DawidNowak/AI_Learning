import os
import io
import sys
import subprocess
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import gradio as gr

GPT_40 = "gpt-4o"
GPT_4O_MINI = "gpt-4o-mini"
# o3-mini is available for organisation tier 3 and up - $100 paid >.>
# GPT_O3_MINI = "o3-mini"
GPT_O1 = "o1"

DEEPSEEK_URL = "https://api.deepseek.com"
DEEPSEEK_R1 = "deepseek-reasoner"
DEEPSEEK_V3 = "deepseek-chat"

CLAUDE_35_SONNET = "claude-3-5-sonnet-latest"
CLAUDE_35_HAIKU = "claude-3-5-haiku-latest"

load_dotenv()

openai = OpenAI()
claude = anthropic.Anthropic()
deepseekai = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url=DEEPSEEK_URL
)

def system_prompt_for(language):
    system_message = f"You are an assistant that reimplements Python code in high performance {language} for a Windows 11. "
    system_message += f"Respond only with {language} code; use comments sparingly and do not provide any explanation other than occasional comments. "
    system_message += f"The {language} response needs to produce an identical output in the fastest possible time."
    return system_message

def user_prompt_for(python, language):
    user_prompt = f"Rewrite this Python code in {language} with the fastest possible implementation that produces identical output in the least time. "
    user_prompt += f"Respond only with {language} code; do not explain your work other than a few comments. "
    user_prompt += f"Pay attention to number types to ensure no int overflows. Remember to include all necessary {language} packages.\n\n"
    user_prompt += python
    return user_prompt

def messages_for(python, language):
    return [
        {"role": "system", "content": system_prompt_for(language)},
        {"role": "user", "content": user_prompt_for(python, language)}
    ]

languages = {
    "C++": {"abbrev": "cpp", "ext": "cpp"},
    "C#": {"abbrev": "csharp", "ext": "cs"},
    "Java": {"abbrev": "java", "ext": "java"},
    "JavaScript": {"abbrev": "javascript", "ext": "js"},
    "TypeScript": {"abbrev": "typescript", "ext": "ts"}
}

def stream_gpt(python, language, model):
    stream = openai.chat.completions.create(
        model=model,
        messages=messages_for(python, language),
        stream=True
    )
    reply = ""
    for chunk in stream:
        fragment = chunk.choices[0].delta.content or ""
        reply += fragment
        yield reply
    write_output(reply, language)

def stream_claude(python, language, model):
    result = claude.messages.stream(
        model=model,
        max_tokens=2000,
        system=system_prompt_for(language),
        messages=[{"role": "user", "content": user_prompt_for(python, language)}]
    )
    reply = ""
    with result as stream:
        for text in stream.text_stream:
            reply += text
            yield reply
    write_output(reply, language)

def stream_deepseek(python, language, model):
    stream = deepseekai.chat.completions.create(
        model=model,
        messages=messages_for(python, language),
        stream=True
    )
    reply = ""
    for chunk in stream:
        fragment = chunk.choices[0].delta.content or ""
        reply += fragment
        yield reply
    write_output(reply, language)

def generate(python, language, model):
    if model in [GPT_40, GPT_4O_MINI, GPT_O1]:
        result = stream_gpt(python, language, model)
    elif model in [CLAUDE_35_SONNET, CLAUDE_35_HAIKU]:
        result = stream_claude(python, language, model)
    elif model in [DEEPSEEK_R1, DEEPSEEK_V3]:
        result = stream_deepseek(python, language, model)
    else:
        raise ValueError("Unknown model")
    for stream_so_far in result:
        yield stream_so_far

def write_output(code: str, language: str):
    lang = languages[language]
    abbrev = lang["abbrev"]
    code = code.replace(f"```{abbrev}","").replace("```","")
    with open(f"./Outputs/{abbrev}.{lang["ext"]}", "w") as file:
        file.write(code)

def execute_python(code):
    try:
        output = io.StringIO()
        sys.stdout = output
        exec(code)
    finally:
        sys.stdout = sys.__stdout__
    return output.getvalue()

def execute(lang: str):
    match lang:
        case "C++":
            return execute_cpp()

def execute_cpp():
    # Install MinGW-w64 from Mingw-w64 downloads.
    # Add the MinGW bin folder to your system PATH
    # Example: C:\mingw-w64\bin
    # Verify installation: g++ --version
    # Compile using MinGW, et voilà
    cpp = languages["C++"]
    source = f"./Outputs/{cpp['abbrev']}.{cpp['ext']}"
    exe = f"./Outputs/{cpp['abbrev']}.exe"
    print(source)
    try:
        compile_cmd = ["g++", "-O3", "-ffast-math", "-std=c++17", "-o", exe, source]
        subprocess.run(compile_cmd, check=True, text=True, capture_output=True)
        run_cmd = [exe]
        run_result = subprocess.run(run_cmd, check=True, text=True, capture_output=True)
        return run_result.stdout
    except subprocess.CalledProcessError as e:
        return f"An error occurred:\n{e.stderr}"

pi = """
import time

def calculate(iterations, param1, param2):
    result = 1.0
    for i in range(1, iterations+1):
        j = i * param1 - param2
        result -= (1/j)
        j = i * param1 + param2
        result += (1/j)
    return result

start_time = time.time()
result = calculate(100_000_000, 4, 1) * 4
end_time = time.time()

print(f"Result: {result:.12f}")
print(f"Execution Time: {(end_time - start_time):.6f} seconds")
"""

with gr.Blocks() as ui:
    with gr.Row():
        python = gr.Textbox(label="Python code:", lines=10, value=pi)
        generated_code = gr.Textbox(label="Generated code:", lines=10)
    with gr.Row():
        model = gr.Dropdown(
            [GPT_40, GPT_4O_MINI, GPT_O1, CLAUDE_35_SONNET, CLAUDE_35_HAIKU, DEEPSEEK_R1, DEEPSEEK_V3],
            label="Select model",
            value=GPT_4O_MINI)
        language = gr.Dropdown(
            languages.keys(),
            label="Select Language",
            value="C#"
        )
        convert = gr.Button("Convert code", variant='primary')
    with gr.Row():
        run_python = gr.Button("Run python", variant='secondary')
        run_translated = gr.Button("Run translated", variant='secondary')
    with gr.Row():
        python_out = gr.TextArea(label="Python result:")
        translated_out = gr.TextArea(label="Translated result:")

    convert.click(
        generate,
        inputs=[python, language, model],
        outputs=[generated_code]
    )

    run_python.click(
        execute_python,
        inputs=[python],
        outputs=[python_out]
    )

    run_translated.click(
        execute,
        inputs=[language],
        outputs=[translated_out]
    )

ui.launch(inbrowser=True)