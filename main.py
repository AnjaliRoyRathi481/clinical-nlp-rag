import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import ollama

app = FastAPI(title="Clinical NLP RAG API")

df = pd.read_csv("data/processed/cleaned_data.csv")
index = faiss.read_index("data/processed/faiss_index")
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.get("/")
def home():
    return {"message": "API running"}

@app.get("/search")
def search(query: str, top_k: int = 3):
    query_vector = model.encode([query]).astype("float32")
    distances, positions = index.search(query_vector, top_k)
    results = []
    for rank, pos in enumerate(positions[0], 1):
        results.append({
            "rank": rank,
            "text": df.iloc[pos]["context_text"],
            "distance": float(distances[0][rank - 1])
        })
    return {"query": query, "results": results}

@app.get("/ask")
def ask(query: str, top_k: int = 3):
    query_vector = model.encode([query]).astype("float32")
    distances, positions = index.search(query_vector, top_k)

    context = ""
    for pos in positions[0]:
        context += df.iloc[pos]["context_text"] + "\n\n"

    prompt = f"""You are a medical research assistant. Answer the question using ONLY the research papers provided below. If the papers do not contain the answer, say so honestly.

Research papers:
{context}

Question: {query}

Answer:"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response["message"]["content"]

    return {"query": query, "answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

