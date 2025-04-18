import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from embed import VECTOR_STORE_PATH



def load_embeddings(path=VECTOR_STORE_PATH):
    with open(path, "rb") as f:
        embeddings, texts = pickle.load(f)
    return embeddings, texts


def retrieve(query, embedding_model, k=3):
    query_embedding = embedding_model.encode([query])[0]
    embeddings, texts = load_embeddings()

    sims = cosine_similarity([query_embedding], embeddings)[0]
    top_k_indices = sims.argsort()[-k:][::-1]
    return [texts[i] for i in top_k_indices]
