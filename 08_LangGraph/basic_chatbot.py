from io import BytesIO
from dotenv import load_dotenv

from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic

from PIL import Image

# Add ANTHROPIC_API_KEY to your .env file
load_dotenv()

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# OR use set_entry_point and set_finish_point
# graph_builder.set_entry_point("chatbot")
# graph_builder.set_finish_point("chatbot")

graph = graph_builder.compile()

# Display the graph image, change to False if you don't want this
if True:
    try:
        image_bytes = graph.get_graph().draw_mermaid_png()
        graph_image = Image.open(BytesIO(image_bytes))
        graph_image.show()
    except Exception as e:
        print(f'Error: Could not display graph image. {e}')
        pass

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
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
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break