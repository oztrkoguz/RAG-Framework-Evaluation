import time
start_time = time.time()

import os
import chromadb
from swarms import Agent, OpenAIChat
from swarms.utils.data_to_text import data_to_text
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Create a custom embedding model using HuggingFace
custom_embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Get the OpenAI API key from environment variables
api_key = os.environ.get("OPENAI_API_KEY")

# Initialize the OpenAIChat model with the specified parameters
llm = OpenAIChat(
    model='gpt-3.5-turbo', openai_api_key=api_key
)

# Create the Agent with the language model and other configurations
agent = Agent(
    llm=llm, 
    max_loops=1, 
    autosave=True, 
    dashboard=True,
    system_prompt="Use the following context pieces to answer the question at the end. If you don't know the answer, just say you don't know, don't try to make up an answer."
)

# Initialize the Chroma vector store with a persistent storage directory
persist_directory = 'fiance_agent_rag'
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=custom_embedding_model)

# Specify the file path to the PDF document
file_path = r"C:\Projects\llm\rag_project\document.pdf"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Load and split the document into text chunks
contract = data_to_text(file_path)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_text(contract)

# Add the text chunks to the vector store
vectorstore.add_texts(texts)

# Prepare the query
query = "How many towers and bastions does Alanya Castle consist of"

# Retrieve the most similar documents from the vector store
docs = vectorstore.similarity_search(query, k=2)

# Create the context from the retrieved documents
context = "\n\n".join([doc.page_content for doc in docs])

# Run the agent with the context and query
agent.run(f"{context}\n\nQuestion: {query}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
