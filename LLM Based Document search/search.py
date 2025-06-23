import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")

def load_faiss_index():
    index = faiss.read_index("C:/Users/Administrator/Desktop/LLM Based Document search/faiss_index/index.faiss")
    with open("faiss_index/chunks.txt", "r", encoding="utf-8") as f:
        chunks = [line.strip() for line in f.readlines()]
    return index, chunks

def embed_query(query):
    return model.encode([query]).astype("float32")

def search(query, k=5):
    index, chunks = load_faiss_index()
    query_vec = embed_query(query)
    distances, indices = index.search(query_vec, k)
    results = [chunks[i] for i in indices[0]]
    return results

if __name__ == "__main__":
    print("Document Semantic Search")
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        results = search(query)
        print("\nTop relevant document chunks:\n" + "-"*40)
        for i, res in enumerate(results, 1):
            print(f"\nResult {i}:\n{res}")
            print("-"*40)
