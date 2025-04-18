# 🧠 Sample RAG (Retrieval-Augmented Generation) Project

This is a simple Retrieval-Augmented Generation (RAG) application that uses:

- 🔍 `all-MiniLM-L6-v2` from `sentence-transformers` for generating dense embeddings
- 🗃️ A local vector store (using the FAISS library) to store document embeddings
- 🤖 `phi4` model served locally via [Ollama](https://ollama.com/) for question-answering

---

## 🚀 Features

- Embed documents and store in a vector DB
- Perform similarity search using a query
- Use relevant chunks as context to query a local LLM (phi4)
- Serve an API via FastAPI

---

## 🛠️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/YashKushwaha/SAMPLE_RAG_PROJECT.git
cd sample-rag-project
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Ensure Ollama is running locally

Install [Ollama](https://ollama.com/) and run the `phi4` model:

```bash
ollama run phi4
```

---

## 📂 Folder Structure

```
sample-rag-project/
│
├── data/                  # Your raw documents
├── vectorstore/           # Pickle file storing embedded vectors
├── src/
│   ├── api.py             # FastAPI server
│   ├── embed.py           # Embedding logic
│   ├── ingest.py          # Load + embed + save vectors
│   ├── retriever.py       # Vector search
│   ├── rag_chain.py       # Combines retrieval and LLM
│   └── utils.py           # Utility functions
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🧪 Running the API

Start the FastAPI server:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Test with Postman or `curl`:

**Endpoint:** `POST http://localhost:8000/ask`  
**Body (JSON):**

```json
{
  "question": "What is this project about?"
}
```

---

## 📈 Planned Improvements

- Add document chunking strategies
- Include file upload for dynamic ingestion
- Add frontend UI (e.g., Streamlit or React)

---

## 📚 References

- [SentenceTransformers](https://www.sbert.net/)
- [Ollama](https://ollama.com/)
- [FastAPI](https://fastapi.tiangolo.com/)

---

## 🧑‍💻 Author

Built with ❤️ by [Yash]
