import os
from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Load API key
load_dotenv()

# Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "web_assistant"

# Globals
llm = None
vector_store = None


# -------------------------------
# PROCESS URLS
# -------------------------------
def process_urls(urls):
    global llm, vector_store

    yield "Initializing LLM..."
    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=1000
        )

    yield "Creating vector database..."

    if not VECTORSTORE_DIR.exists():
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR)
    )

    yield "Loading data from URLs..."

    loader = WebBaseLoader(urls)
    loader.requests_kwargs = {"timeout": 10}
    documents = loader.load()

    yield "Splitting documents..."

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE
    )
    docs = splitter.split_documents(documents)

    yield "Adding to vector DB..."

    ids = [str(uuid4()) for _ in docs]
    vector_store.add_documents(docs, ids=ids)

    yield "Vector DB ready ✅"


# -------------------------------
# GENERATE ANSWER
# -------------------------------
def generate_answer(query):
    if vector_store is None:
        raise RuntimeError("Vector DB not initialized")

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(query)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are a helpful assistant.
Answer ONLY using the content below.
If answer is not found, say "I don't know".

Content:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    sources = "\n".join(
        set(doc.metadata.get("source", "") for doc in docs)
    )

    return response.content.strip(), sources