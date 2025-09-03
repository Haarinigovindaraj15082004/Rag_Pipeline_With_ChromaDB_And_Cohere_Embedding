# ================================
# 0. IMPORTS AND SETUP
# ================================

import os
import hashlib
from typing import List
import cohere
import chromadb
from groq import Groq
from dotenv import load_dotenv

# Unstructured.io for advanced document parsing
from unstructured.partition.auto import partition
from unstructured.chunking.basic import chunk_elements

# Pandas for structured files
import pandas as pd

# Load API keys from .env
load_dotenv()

# Initialize APIs
co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ================================
# 1. CHROMA CLOUD SETUP
# ================================

client = chromadb.CloudClient(
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
    api_key=os.getenv("CHROMA_API_KEY")
)

# Create or get collection (like a table in DB)
collection = client.get_or_create_collection(name="rag_knowledgebase")

# ================================
# 2. FILE EXTRACTION
# ================================

#unstructured files
def extract_text_with_unstructured(file_path: str) -> List[str]:
    """
    Extract and chunk text using unstructured.io
    Works for PDF, DOCX, TXT, PPTX, etc.
    """
    elements = partition(filename=file_path)
    chunks = chunk_elements(elements, max_characters=1000, overlap=100)
    return [str(chunk) for chunk in chunks]

#structured files
def extract_structured_data(file_path: str) -> List[str]:
    """
    Extract row-wise and column-wise text from CSV/XLSX
    """
    df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)
    
    # Row-wise
    row_texts = df.astype(str).apply(lambda x: " | ".join(x), axis=1).tolist()
    
    # Column-wise (first 10 unique values per column)
    col_texts = [f"{col}: {', '.join(df[col].dropna().astype(str).unique()[:10])}" 
                 for col in df.columns]
    
    return row_texts + col_texts

# ================================
# 3. DEDUPLICATION & HASHING
# ================================
#hashlib.md5 → takes text and converts it into a fixed-length code.
def get_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def is_duplicate(text: str) -> bool:
    h = get_hash(text)
    existing = collection.get(where={"hash": h})
    return len(existing['ids']) > 0

# ================================
# 4. INGEST FILES INTO CHROMA
# ================================

def ingest_file(file_path: str):
    print(f"📄 Processing {file_path}...")
    
    # Extract text
    if file_path.endswith((".csv", ".xlsx")):
        chunks = extract_structured_data(file_path)
    else:
        chunks = extract_text_with_unstructured(file_path)

    new_texts, new_hashes, new_ids = [], [], []
    
    for i, chunk in enumerate(chunks):
        if not is_duplicate(chunk):
            h = get_hash(chunk)
            new_texts.append(chunk)
            new_hashes.append(h)
            new_ids.append(f"{file_path}_{i}"[:50])  # ID limit in ChromaDB
    
    if new_texts:
        embeddings = co.embed(
            texts=new_texts,
            model="embed-english-v3.0",
            input_type="search_document"
        ).embeddings

        # Add to ChromaDB
        collection.add(
            embeddings=embeddings,
            documents=new_texts,
            metadatas=[{"source": file_path, "hash": h} for h in new_hashes],
            ids=new_ids
        )
        print(f"✅ Added {len(new_texts)} new chunks from {file_path}")
    else:
        print(f"⏭️  No new content to add from {file_path}")

# ================================
# 5. TOP-K RETRIEVAL
# ================================

def retrieve_context(query: str, k: int = 3) -> List[str]:
    #Convert query into an embedding
    query_emb = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query"
    ).embeddings[0]
    
    #semantic search
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=k,
        include=["documents", "metadatas"]
    )
    
    #Extract retrieved docs and metadata
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return [f"[{meta.get('source','')}] {doc}" for doc, meta in zip(docs, metas)]

# ================================
# 6. LLM GENERATION (GROQ LLAMA-3)
# ================================

def generate(prompt: str, model: str = "llama-3.1-8b-instant", max_tokens: int = 1024) -> str:
    completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.3,
        max_tokens=max_tokens
    )
    return completion.choices[0].message.content

# ================================
# 7. RAG TASKS
# ================================

def rag_query(question: str, k: int = 3) -> str:
    context_chunks = retrieve_context(question, k)
    context_text = "\n\n---\n\n".join(context_chunks)
    prompt = f"""
Use only the following context to answer the question.
If the answer is not in the context, say 'I don't know'.

Context:
{context_text}

Question: {question}
Answer:
"""
    return generate(prompt)

# ================================
# 8. USAGE EXAMPLE
# ================================

if __name__ == "__main__":
    # Ingest files (supports PDF, DOCX, TXT, CSV, XLSX)
    ingest_file("iv_list.pdf")

    # Query
    user_q = input("❓ Ask your question: ")
    answer = rag_query(user_q)
    print("\n💡 Answer:", answer)
