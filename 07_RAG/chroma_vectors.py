import os
import glob
import shutil
from dotenv import load_dotenv

# pip install langchain-community --user
# pip install langchain --user
# pip install langchain-openai --user
# pip install langchain-chroma --user
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# pip install numpy --user
# pip install scikit-learn --user
# pip install plotly --user
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go

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

print(f"Loaded {len(documents)} documents and split into {len(chunks)} chunks.")

# The unique document types - the folder names
doc_types = set(chunk.metadata[DOC_TYPE] for chunk in chunks)

# Delete the existing vectorstore if it exists
if os.path.exists(DB_NAME):
    shutil.rmtree(DB_NAME)

# Create a new Chroma Vectorstore
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_NAME)
print(f"Vectorstore created with {vectorstore._collection.count()} chunks")

# Get one vector and find how many dimensions it has
collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"The vectors have {dimensions:,} dimensions")

# Visualize the vectorstore

# Prework
result = collection.get(include=['embeddings', 'documents', 'metadatas'])
vectors = np.array(result['embeddings'])
documents = result['documents']
doc_types = [metadata['doc_type'] for metadata in result['metadatas']]
colors = [['blue', 'green', 'red', 'orange'][['products', 'employees', 'contracts', 'company'].index(t)] for t in doc_types]

# Reduce the dimensionality of the vectors to 2D using t-SNE
# (t-distributed stochastic neighbor embedding)
tsne = TSNE(n_components=2, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

# Create the 2D scatter plot
fig = go.Figure(data=[go.Scatter(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])

fig.update_layout(
    title='2D Chroma Vector Store Visualization',
    scene=dict(xaxis_title='x',yaxis_title='y'),
    width=800,
    height=600,
    margin=dict(r=20, b=10, l=10, t=40)
)

fig.show()

# Reduce the dimensionality of the vectors to 3D using t-SNE
# (t-distributed stochastic neighbor embedding)
tsne = TSNE(n_components=3, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

# Create the 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    z=reduced_vectors[:, 2],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])

fig.update_layout(
    title='3D Chroma Vector Store Visualization',
    scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
    width=900,
    height=700,
    margin=dict(r=20, b=10, l=10, t=40)
)

fig.show()