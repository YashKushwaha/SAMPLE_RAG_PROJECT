import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from embed import VECTOR_STORE_PATH, TEXT_STORE_PATH
import faiss


def load_embeddings_old(path=VECTOR_STORE_PATH):
    with open(path, "rb") as f:
        embeddings, texts = pickle.load(f)
    return embeddings, texts

def load_embeddings(index_path=VECTOR_STORE_PATH, text_path=TEXT_STORE_PATH):
    # Load FAISS index
    index = faiss.read_index(index_path)

    # Load associated texts
    with open(text_path, "rb") as f:
        texts = pickle.load(f)

    return index, texts

def retrieve(query, embedding_model, k=3):
    # Generate the embedding for the query
    query_embedding = embedding_model.encode([query])[0]
    query_embedding = np.array([query_embedding]).astype("float32")

    # Load FAISS index and corresponding texts
    index, texts = load_embeddings()

    # Perform search in FAISS
    D, I = index.search(query_embedding, k)  # I is indices of top-k results

    # Fetch and return corresponding texts
    return [texts[i] for i in I[0]]