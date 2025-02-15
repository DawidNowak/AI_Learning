import os
import glob
from dotenv import load_dotenv
import gradio as gr
import anthropic

MODEL = "claude-3-5-sonnet-20241022"

load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

claude = anthropic.Anthropic(api_key=anthropic_api_key)

context = {}

employees = glob.glob("simple_rag_knowledge_base/employees/*")

for employee in employees:
    name = employee.split(' ')[-1][:-3]
    doc = ""
    with open(employee, "r", encoding="utf-8") as f:
        doc = f.read()
    context[name]=doc

products = glob.glob("simple_rag_knowledge_base/products/*")

for product in products:
    name = product.split(os.sep)[-1][:-3]
    doc = ""
    with open(product, "r", encoding="utf-8") as f:
        doc = f.read()
    context[name]=doc

system_message = ("You are an expert in answering accurate questions about Insurellm, the Insurance Tech company. "
                  "Give brief, accurate answers. If you don't know the answer, say so. "
                  "Do not make anything up if you haven't been provided with relevant context.")

def get_relevant_context(message):
    relevant_context = []
    for context_title, context_details in context.items():
        if context_title.lower() in message.lower():
            relevant_context.append(context_details)
    return relevant_context

def add_context_to_message(message):
    relevant_context = get_relevant_context(message)
    if relevant_context:
        message += "\n\nThe following additional context might be relevant in answering this question:\n\n"
        for relevant in relevant_context:
            message += relevant + "\n\n"
    else:
        message += "\n\nNo additional context is available for this question.\n\n"
    return message


# Ask about any of the employees or products in the knowledge base
# For example, "Who is Alex Chen?" or "What is Carllm?"
def chat(message, history):
    if len(history) == 0:
        history = [{"role": "user", "content": system_message}]
    messages = []
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    message = add_context_to_message(message)
    print(message)
    messages.append({"role": "user", "content": message})

    resp = ""
    with claude.messages.stream(
        model=MODEL,
        max_tokens=200,
        messages=messages
    ) as stream:
        for text in stream.text_stream:
            resp = resp + text
            yield resp

view = gr.ChatInterface(chat, type="messages").launch()