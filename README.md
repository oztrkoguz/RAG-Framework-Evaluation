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
