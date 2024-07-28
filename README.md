## RAG-Framework-Evaluation

![workflow2](https://github.com/user-attachments/assets/c766cd45-2b55-41db-9929-a6c0d1fae8e7)

#### This project aims to compare different Retrieval-Augmented Generation (RAG) frameworks. Using the same document and model, we evaluate LlamaIndex, Autogen, Langchain, Swarms and Crewai frameworks in terms of speed, accuracy and performance. These evaluations will help determine which framework is more effective in certain scenarios.




|       | Prompt Template | Model | Embbeding Model | Vectore Store | Chunk Size | Chunk Overlap |
|-------|-----------------|-------|-----------------|---------------|------------|---------------|
| Fixed | Use the following context pieces to answer the question at the end.<br> If you don't know the answer, just say you don't know, don't try to make up an answer.<br> Give answers in great detail. |gpt-3.5-turbo|BAAI/bge-small-en-v1.5|Chroma|1000|100|


| Framework | Time |Easy Integration|
|----------|----------|-------------|
| Autogen  | 12.68s  | + |
| Crewai   | 17.76s   | - |
| Langchain  | 12.18s  | + |
| Llamaindex  | 12.44s  | - |
| Swarms  | 17.30s  | + |


### Usage
```
git clone https://github.com/oztrkoguz/RAG-Framework-Evaluation.git
cd RAG-Framework-Evaluation
#autogen
python rag_autogen.py
#crewai
python rag_crewai.py
#langchain
python rag_langchain.py
#llamaindex
python rag_llamaindex.py
#swarms
python rag_swarm.py
```

### Requirements
```
autogen==1.0.16
crewai==0.41.1
crewai-tools==0.4.26
langchain==0.1.20
langchain-chroma==0.1.2
langchain-cohere==0.1.9
langchain-community==0.0.38
langchain-core==0.1.52
langchain-experimental==0.0.62
langchain-openai==0.1.17
langchain-text-splitters==0.0.2
langsmith==0.1.93
llama-index==0.10.56
llama-index-agent-openai==0.2.9
llama-index-cli==0.1.12
llama-index-core==0.10.56
llama-index-embeddings-huggingface==0.2.2
llama-index-embeddings-langchain==0.1.2
llama-index-embeddings-openai==0.1.11
llama-index-indices-managed-llama-cloud==0.2.5
llama-index-legacy==0.9.48
llama-index-llms-anyscale==0.1.4
llama-index-llms-fireworks==0.1.5
llama-index-llms-langchain==0.1.4
llama-index-llms-openai==0.1.26
llama-index-multi-modal-llms-openai==0.1.8
llama-index-program-openai==0.1.6
llama-index-question-gen-openai==0.1.3
llama-index-readers-file==0.1.30
swarms==5.4.0
swarms-memory==0.0.2

```
