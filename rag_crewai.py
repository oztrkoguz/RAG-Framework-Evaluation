import time
start_time = time.time()

from crewai import Agent, Task, Crew
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool

# Load and process PDF
loader = PyPDFLoader("document.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Create embeddings and Chroma vector database
custom_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = Chroma.from_documents(docs, custom_embeddings)

# PDF search tool
def search_pdf(query):
    docs = vectorstore.similarity_search(query, k=1)
    return docs[0].page_content if docs else "No results found."

pdf_tool = Tool(
    name="PDF Search",
    func=search_pdf,
    description="Used to search within a PDF document."
)

data_analyst = Agent(
    role='Data Analyst',
    goal='Analyze the information in the PDF document and extract key insights',
    backstory='You are an experienced data analyst specialized in analyzing complex data and extracting meaningful information. You have the ability to quickly and accurately process information from PDF documents.',
    verbose=True,
    allow_delegation=False,
    tools=[pdf_tool]
)

test_task = Task(
    description="'document.pdf' file analysis. How many bastions and towers does Alanya Castle consist of?",
    expected_output="Use the following context pieces to answer the question at the end.If you don't know the answer, just say you don't know, don't try to make up an answer. Give answers in great detail.",
    agent=data_analyst
)

crew = Crew(
    agents=[data_analyst],
    tasks=[test_task],
    verbose=2  # Detailed output
)

result = crew.kickoff()
print(result)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
