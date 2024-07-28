import time
start_time = time.time()
import os
from llama_index.core import Prompt, Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# PDF file path
pdf_path = 'document.pdf'

# PDF loader settings
filename_fn = lambda filename: {'file_name': os.path.basename(pdf_path)}
loader = SimpleDirectoryReader(input_files=[pdf_path], file_metadata=filename_fn)
documents = loader.load_data()

# Embedding model
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# LLM (Language Model) settings
llm = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='gpt-3.5-turbo'
)

# Create Chroma vector store
vector_store = Chroma(embedding_function=embed_model)

# Configure settings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 1000
Settings.chunk_overlap = 100

# Convert the document to a vector store index
index = VectorStoreIndex.from_documents(
    documents, 
    embed_model=embed_model,
    llm=llm,
    vector_store=vector_store
)

# Question-answer template
text_qa_template = Prompt("""
[INST] <>
If you don't know the answer, just say you don't know, don't try to make up an answer. Give answers in great detail.
<>
Refer to the following Consultation Guidelines and example consultations: {context_str}
Continue the conversation: {query_str}
""")

# Query Engine
query_engine = index.as_query_engine(text_qa_template=text_qa_template, streaming=True, llm=llm)

# Single inference
def single_inference(query):
    messages = [
        HumanMessage(content=query)
    ]
    # Use the synchronous method for generating a response
    response = llm(messages=messages)
    
    # Extract content from response
    response_content = response.content if isinstance(response, AIMessage) else "No valid response received."
    
    print(response_content)

# Example usage
user_query = "How many towers and bastions does Alanya Castle consist of?"

# Run the inference
single_inference(user_query)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTotal execution time: {elapsed_time:.2f} seconds")