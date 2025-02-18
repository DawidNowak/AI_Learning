import os
import glob
import shutil
from dotenv import load_dotenv
import gradio as gr
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

MODEL = "gpt-4o-mini"
DB_NAME = "vector_db"

DOC_TYPE = "doc_type"

load_dotenv()
embeddings = OpenAIEmbeddings()

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

# Get one vector and find how many dimensions it has
collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"Dimensions: {dimensions}")

# create a new Chat with OpenAI
llm = ChatOpenAI(temperature=0.7, model_name=MODEL)

# set up the conversation memory for the chat
# This line produces LangChainDeprecationWarning: Please see the migration guide at: https://python.langchain.com/docs/versions/migrating_memory/
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# the retriever is an abstraction over the VectorStore that will be used during RAG
retriever = vectorstore.as_retriever()

# putting it together: set up the conversation chain with the GPT 4o-mini LLM, the vector store and memory
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# Wrapping in a function - note that history isn't used, as the memory is in the conversation_chain
def chat(message, history):
    result = conversation_chain.invoke({"question": message})
    return result["answer"]

view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True)