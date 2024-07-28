import time
start_time = time.time()

import os
from langchain.chat_models import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# Custom adapter class for LlamaIndex embeddings to make it compatible with LangChain
class LlamaIndexEmbeddingAdapter(Embeddings):
    def __init__(self, llama_index_embedding):
        self.llama_index_embedding = llama_index_embedding

    def embed_documents(self, texts):
        return self.llama_index_embedding.get_text_embedding_batch(texts)

    def embed_query(self, text):
        return self.llama_index_embedding.get_text_embedding(text)

# Initialize the OpenAI language model with the provided API key
llm = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='gpt-3.5-turbo'
)

# Load the PDF document
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# Split the documents into chunks for processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

# Create the LlamaIndex embedding model
llama_index_embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Use the adapter class to create a LangChain-compatible embedding model
embed_model = LlamaIndexEmbeddingAdapter(llama_index_embed_model)

# Create a vector store from the document splits and embeddings
vectorstore = Chroma.from_documents(documents=splits, embedding=embed_model)
retriever = vectorstore.as_retriever()

# Define a custom RAG prompt template
template = """Use the following context pieces to answer the question at the end.
If you don't know the answer, just say you don't know, don't try to make up an answer. Give answers in great detail.
{context}
Question: {question}
Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

# Function to format documents for the RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the RAG chain with retriever, prompt template, and language model
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt 
    | llm
    | StrOutputParser()
)

# Invoke the RAG chain with a specific question
response = rag_chain.invoke("How many towers and bastions does Alanya Castle consist of")
print(response)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
