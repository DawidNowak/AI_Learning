from dotenv import load_dotenv
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

from langgraph.checkpoint.memory import MemorySaver

# Add ANTHROPIC_API_KEY to your .env file
load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")

memory = MemorySaver()

# Compile the graph with the provided checkpointer
graph = graph_builder.compile(checkpointer=memory)

# Display the graph image, change to False if you don't want this
if True:
    from PIL import Image
    from io import BytesIO

    try:
        image_bytes = graph.get_graph().draw_mermaid_png()
        graph_image = Image.open(BytesIO(image_bytes))
        graph_image.show()
    except Exception as e:
        print(f'Error: Could not display graph image. {e}')
        pass

# Pick a thread to use as the key for this conversation.
config = {"configurable": {"thread_id": "1"}}

def stream_graph_updates(user_input: str):
    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config
    ):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        break