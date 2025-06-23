#Needs fixing (may be need more data, or check the code, meanwhile use python search.py)

import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = SentenceTransformer("all-MiniLM-L6-v2")

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
generator = AutoModelForCausalLM.from_pretrained("distilgpt2")

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

def generate_answer(context_chunks, question, max_length=150):
    context = "\n".join(context_chunks)
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"

    inputs = tokenizer.encode(prompt, return_tensors="pt")

    max_model_length = 1024  # distilgpt2 max input length

    if inputs.size(1) > max_model_length:
        inputs = inputs[:, -max_model_length:]

    outputs = generator.generate(
        inputs,
        max_length=inputs.size(1) + max_length,  # total max length = input + generated tokens
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=torch.ones_like(inputs)  
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer[len(prompt):].strip()

if __name__ == "__main__":
    print("Document Semantic Search + Simple Answer Generation 🔎")
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        
        print("\nSearching relevant document chunks...")
        results = search(query)
        
        print("\nGenerating answer...")
        answer = generate_answer(results, query)
        
        print("\nAnswer:\n" + "-"*40)
        print(answer)
        print("-"*40)
