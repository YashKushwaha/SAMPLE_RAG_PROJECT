### rag_chain.py
from retriever import retrieve
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi4"


def query_ollama(prompt, model=MODEL):
    response = requests.post(OLLAMA_URL, json={"model": model, "prompt": prompt, "stream" : False})
    response.raise_for_status()
    return response.json().get("response")


def run_rag_pipeline(user_query, embedding_model):
    relevant_chunks = retrieve(user_query, embedding_model)
    context = "\n".join(relevant_chunks)
    prompt = f"Answer the question using the context below:\nContext:\n{context}\n\nQuestion: {user_query}"
    return query_ollama(prompt)