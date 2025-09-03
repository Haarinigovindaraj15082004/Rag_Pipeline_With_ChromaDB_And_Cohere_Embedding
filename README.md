# 📚 RAG Pipeline with ChromaDB, Cohere & Groq

This project implements a **Retrieval-Augmented Generation (RAG)** system that extracts knowledge from structured (CSV, Excel) and unstructured (PDF, DOCX, TXT) files, stores it in **ChromaDB**, and answers user questions using **semantic search** and **Groq’s Llama 3 model**.

---

## 🚀 Features
- Extracts text from:
  - **Unstructured files** → PDF, DOCX, TXT (via Unstructured.io)
  - **Structured files** → CSV, Excel (via Pandas)
- **Deduplication** → Prevents storing duplicate chunks using MD5 hashing.
- **Semantic Search** → Finds the most relevant chunks using Cohere embeddings + ChromaDB.
- **LLM Integration** → Uses Groq (Llama 3) to generate answers only from retrieved context.
- **User Input Querying** → Ask any question from ingested documents.

---

## ⚙️ Tech Stack
- **Python 3.13**
- **ChromaDB (Cloud/Local)** → Vector database
- **Cohere API** → Embeddings
- **Groq API** → LLM responses
- **Unstructured.io** → Document parsing
- **Pandas** → Structured data handling

---

## 📂 Project Structure
```

rag-project/
│── rag\_app.py         # Main RAG pipeline
│── requirements.txt   # Dependencies
│── .gitignore         # Ignore env/venv/secrets
│── .env               # API keys (not committed)
│── iv\_list.pdf        # Sample file to ingest

````

---

## 🔑 Setup

1. Clone repo:
   ```bash
   git clone https://github.com/<your-username>/rag-project.git
   cd rag-project
````

2. Create virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Add `.env` file:

   ```
   COHERE_API_KEY=your_cohere_key
   GROQ_API_KEY=your_groq_key
   CHROMA_TENANT=your_chroma_tenant
   CHROMA_DATABASE=your_chroma_database
   CHROMA_API_KEY=your_chroma_api_key
   ```

---

## ▶️ Usage

1. Ingest documents:

   ```bash
   python rag_app.py
   ```

2. Enter a query:

   ```
   ❓ Ask your question: What are the main findings in the report?
   ```

3. Get an AI-generated answer based only on your documents:

   ```
   💡 Answer: The report highlights an improvement in customer satisfaction during Q2 due to better service response times.
   ```

---

## 📊 How It Works

1. **Extraction** → Text is extracted & chunked.
2. **Embedding** → Each chunk is embedded using Cohere.
3. **Storage** → Stored in ChromaDB with metadata & hash.
4. **Querying** → User query is embedded, ChromaDB retrieves top-k similar chunks.
5. **Generation** → Groq (Llama 3) uses retrieved chunks to answer.

---

## 🛡️ Notes

* `.env` file is ignored in Git (`.gitignore`) → never push API keys.
* Supports **both structured & unstructured data**.
* Default retrieval: **Top-3 results (k=3)**.

---

## ✨ Example

Input:

```
❓ Ask your question: Why did customer satisfaction improve in Q2?
```

Output:

```
💡 Answer: Customer satisfaction improved in Q2 due to faster service response times and proactive customer engagement.
```

---
