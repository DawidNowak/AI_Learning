import asyncio
import os
import glob
import shutil
import gradio as gr
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.tools import tool
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

MODEL = "gpt-4o-mini"
DB_NAME = "vector_db"

DOC_TYPE = "doc_type"

load_dotenv()
embeddings = OpenAIEmbeddings()

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# Knowledge base comes from a course - LLM engineering by Edward Donner https://edwarddonner.com/
# https://www.udemy.com/course/llm-engineering-master-ai-and-large-language-models/?referralCode=35EB41EBB11DD247CF54

folders = glob.glob("knowledge_base/*")
text_loader_kwargs = {'encoding': 'utf-8'}

# All documents in the knowledge base
documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    folder_docs = loader.load()
    for doc in folder_docs:
        # Add document type based on the folder name to the metadata
        doc.metadata[DOC_TYPE] = doc_type
        documents.append(doc)

# Split documents into chunks of approximately 1000 characters with 200 characters overlap
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# The unique document types - the folder names
doc_types = set(chunk.metadata[DOC_TYPE] for chunk in chunks)

# Delete the existing vectorstore if it exists
if os.path.exists(DB_NAME):
    shutil.rmtree(DB_NAME)

# Create a new Chroma Vectorstore
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_NAME)
retriever = vectorstore.as_retriever()

@tool
def chroma_retrieval_tool(query: str) -> str:
    """Get relevant information from the knowledge base using retriever."""
    docs = asyncio.run(retriever.ainvoke(query))
    # Return concatenated document content (adjust as needed)
    return "\n\n".join(doc.page_content for doc in docs)

tools = [chroma_retrieval_tool]
tool_node = ToolNode(tools=tools)

# create a new Chat with OpenAI
llm = ChatOpenAI(temperature=0.7, model_name=MODEL).bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.set_entry_point("chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_finish_point("chatbot")

memory = MemorySaver()
# Compile the graph with the provided checkpointer
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

def chat(message, history):
    if history is None:
        history = []
    print(history)
    # Prepare the initial state with the user's message as a dictionary.
    state = {"messages": [{"role": "user", "content": message}]}
    final_state = graph.invoke(state, config=config)
    # Extract the assistant's response using attribute access.
    return final_state["messages"][-1].content

view = gr.ChatInterface(chat, type="messages")
view.launch(inbrowser=True)
