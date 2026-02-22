# Retrieval-Augmented Generation (RAG) with LangChain, Gemini and Pinecone
- Jeimy Alejandra Yaya Martinez

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)**
system using:

-   **Google Gemini (`gemini-2.5-flash`)** as the Large Language Model
    (LLM)
-   **Google Gemini Embeddings (`gemini-embedding-001`)**
-   **Pinecone** as the vector database
-   **LangChain (LCEL)** for pipeline composition

The system enhances LLM responses by retrieving relevant contextual
information from a vector database before generating the final answer.

This repository follows the official LangChain RAG tutorial structure
and demonstrates how retrieval integrates with generative models to
improve factual grounding and contextual accuracy.

------------------------------------------------------------------------

# Architecture

## High-Level Architecture

User Question\
↓\
Retriever (Pinecone)\
↓\
Relevant Context (Vector Similarity Search)\
↓\
Prompt Template\
↓\
Gemini LLM (`gemini-2.5-flash`)\
↓\
Final Answer

------------------------------------------------------------------------

## Components

### 1. Document Loader

-   `WebBaseLoader`
-   Loads content from the LangChain RAG tutorial webpage.

### 2. Text Splitter

-   `RecursiveCharacterTextSplitter`
-   Splits documents into smaller chunks.
-   Chunk size: 1000 characters\
-   Overlap: 200 characters

### 3. Embeddings Model

-   `GoogleGenerativeAIEmbeddings`
-   Model used: `gemini-embedding-001`
-   Converts text chunks into dense vectors (3072 dimensions).

### 4. Vector Database

-   Pinecone (Serverless index)
-   Dimension: 3072
-   Metric: cosine similarity

### 5. Retriever

-   Retrieves relevant document chunks based on vector similarity.

### 6. Prompt Template

    You are an assistant for question-answering tasks.
    Use the retrieved context to answer the question.
    If you don't know the answer, say you don't know.

    Context:
    {context}

    Question:
    {question}

### 7. LLM

-   `ChatGoogleGenerativeAI`
-   Model: `gemini-2.5-flash`
-   Temperature: 0

### 8. LCEL RAG Chain

``` python
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
```
This pipeline ensures:

1. The question is passed to the retriever.
2. Relevant documents are retrieved.
3. Retrieved content is injected into the prompt.
4. Gemini generates the final grounded response.


------------------------------------------------------------------------

# Project Structure

    RAG-project/
    │
    ├── rag.py
    ├── requirements.txt
    ├── .env
    └── README.md

------------------------------------------------------------------------

# Installation Guide

## 1. Clone the Repository

``` bash
git clone https://github.com/your-username/rag-project.git
cd rag-project
```

## 2. Create Virtual Environment

### Windows

``` bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

``` bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

``` bash
pip install -r requirements.txt
```

Dependencies include:

- langchain
- langchain-google-genai
- langchain-pinecone
- pinecone-client
- python-dotenv

------------------------------------------------------------------------

# Environment Variables

Create a `.env` file in the project root:

    GOOGLE_API_KEY=your_gemini_api_key
    PINECONE_API_KEY=your_pinecone_api_key
    USER_AGENT=rag-project

------------------------------------------------------------------------

# Running the Project

``` bash
python rag.py
```

------------------------------------------------------------------------

# Example Output

    Retrieval Augmented Generation (RAG) is a technique used by sophisticated question-answering (Q&A) chatbots...

<img width="642" height="135" alt="image" src="https://github.com/user-attachments/assets/07c9972f-62f5-4ea8-9e62-a3bbefe23a51" />



------------------------------------------------------------------------

# How Retrieval Enhances the Model

Without retrieval:

- The LLM answers based only on its training data.
- May hallucinate.

With RAG:

- The question is converted into embeddings.
- Pinecone retrieves relevant document chunks.
- Retrieved context is injected into the prompt.
- The LLM generates grounded responses.
This reduces hallucinations and improves factual accuracy.

------------------------------------------------------------------------

# Conclusion

This project demonstrates a complete Retrieval-Augmented Generation pipeline using modern LLM engineering practices. It highlights the integration of semantic search and generative AI, showcasing how vector databases enhance language model performance in real-world applications.
