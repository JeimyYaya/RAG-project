import os
from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader

from pinecone import Pinecone

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# 1. Load Documents
# -----------------------------
loader = WebBaseLoader("https://python.langchain.com/docs/tutorials/rag/")
docs = loader.load()

# -----------------------------
# 2. Split Documents
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(docs)

# -----------------------------
# 3. Gemini Embeddings
# -----------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# -----------------------------
# 4. Store in Pinecone
# -----------------------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "rag-gemini-index"

vectorstore = PineconeVectorStore.from_documents(
    documents=splits,
    embedding=embeddings,
    index_name=index_name
)

retriever = vectorstore.as_retriever()

# -----------------------------
# 5. Initialize Gemini LLM
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# -----------------------------
# 6. Prompt Template
# -----------------------------
prompt = ChatPromptTemplate.from_template("""
You are an assistant for question-answering tasks.
Use the retrieved context to answer the question.
If you don't know the answer, say you don't know.

Context:
{context}

Question:
{question}
""")

# -----------------------------
# 7. Build RAG Chain (LCEL)
# -----------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# -----------------------------
# 8. Ask Question
# -----------------------------
response = rag_chain.invoke(
    "What is Retrieval-Augmented Generation?"
)

print(response)