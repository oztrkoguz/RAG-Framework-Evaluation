import time
start_time = time.time()

import os
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from langchain.embeddings import HuggingFaceEmbeddings

# Initialize custom embeddings using HuggingFace model
custom_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Configuration for the language model
llm_config = {
    "timeout": 600,  # Timeout for API requests
    "cache_seed": 42,  # Seed for caching
    "config_list": [{"model": "gpt-3.5-turbo", "api_key": os.environ["OPENAI_API_KEY"]}],  # Model and API key
}

# Initialize the assistant agent with system messages and LLM configuration
assistant = RetrieveAssistantAgent(
    name="assistant",
    system_message="If you don't know the answer, just say you don't know, don't try to make up an answer. Give answers in great detail.",
    llm_config=llm_config,
)

# Initialize the proxy agent for retrieval with configuration settings
ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER", 
    max_consecutive_auto_reply=3, 
    retrieve_config={
        "task": "code",  
        "docs_path": [
            "document.pdf",
            os.path.join(os.path.abspath(""), "..", "website", "docs"),
        ],  
        "custom_text_types": ["non-existent-type"],  
        "chunk_token_size": 1000,  
        "chunk_overlap": 100,  
        "model": llm_config["config_list"][0]["model"],  
        "vector_db": "chroma",  
        "overwrite": True,  
        "embedding_model": custom_embeddings, 
    },
    code_execution_config=False, 
)

# Define the problem to solve
question = "How many towers and bastions does Alanya Castle consist of"

# Initiate chat with the proxy agent to solve the problem
chat_result = ragproxyagent.initiate_chat(
    assistant, message=ragproxyagent.message_generator, problem=question
)

print(chat_result)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
