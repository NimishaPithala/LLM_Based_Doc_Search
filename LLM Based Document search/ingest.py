import os
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# Load sentence transformer (embedding model)
model = SentenceTransformer("all-MiniLM-L6-v2")  # small and fast


def read_pdfs_from_folder(folder_path):
    all_text = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            all_text.append(text)
    return all_text


def chunk_text(text, max_words=200):
    words = text.split()
    chunks = [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    return chunks


def get_embeddings(texts):
    return model.encode(texts)


def store_index(embeddings, texts):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs("faiss_index", exist_ok=True)
    faiss.write_index(index, "faiss_index/index.faiss")

    with open("faiss_index/chunks.txt", "w", encoding="utf-8") as f:
        for chunk in texts:
            f.write(chunk.strip().replace("\n", " ") + "\n")

if __name__ == "__main__":
    print("Reading PDFs")
    docs = read_pdfs_from_folder("C:/Users/Administrator/Desktop/LLM Based Document search/data")

    print("Chunking text")
    chunks = []
    for doc in docs:
        chunks.extend(chunk_text(doc))

    print(f"Total chunks: {len(chunks)}")

    print("Generating embeddings")
    embeddings = get_embeddings(chunks)

    print("Saving FAISS index")
    store_index(np.array(embeddings).astype("float32"), chunks)

    print(" Done")
