import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import streamlit as st
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import ollama

st.set_page_config(page_title="Clinical NLP RAG", page_icon="🩺", layout="centered")

# ---- Custom colours and styling ----
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e0f7fa 0%, #f1f8e9 100%);
}
.main-title {
    text-align: center;
    color: #00695c;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 0px;
}
.subtitle {
    text-align: center;
    color: #455a64;
    font-size: 17px;
    margin-bottom: 30px;
}
.answer-box {
    background-color: #ffffff;
    border-left: 6px solid #00897b;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    color: #263238;
    font-size: 16px;
    line-height: 1.6;
}
.paper-box {
    background-color: #ffffff;
    border-left: 4px solid #26a69a;
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 12px;
    color: #37474f;
    font-size: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.stButton>button {
    background-color: #00897b;
    color: white;
    font-weight: 600;
    border-radius: 8px;
    border: none;
    padding: 10px 30px;
    font-size: 16px;
}
.stButton>button:hover {
    background-color: #00695c;
    color: white;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_everything():
    df = pd.read_csv("data/processed/cleaned_data.csv")
    index = faiss.read_index("data/processed/faiss_index")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return df, index, model

df, index, model = load_everything()

st.markdown('<p class="main-title">🩺 Clinical Research Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Answers grounded in 211,000 PubMed medical research papers</p>', unsafe_allow_html=True)

query = st.text_input("Ask a medical question:")

if st.button("🔍 Ask"):
    if query:
        with st.spinner("Searching papers and generating answer..."):
            query_vector = model.encode([query]).astype("float32")
            distances, positions = index.search(query_vector, 3)

            context = ""
            papers = []
            for pos in positions[0]:
                text = df.iloc[pos]["context_text"]
                context += text + "\n\n"
                papers.append(text)

            prompt = f"""You are a medical research assistant. Answer the question using ONLY the research papers provided below. If the papers do not contain the answer, say so honestly.

Research papers:
{context}

Question: {query}

Answer:"""

            response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
            answer = response["message"]["content"]

        st.markdown("### 💬 Answer")
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

        st.markdown("### 📄 Source Papers")
        for i, paper in enumerate(papers, 1):
            preview = paper[:400] + "..."
            st.markdown(f'<div class="paper-box"><b>Paper {i}</b><br>{preview}</div>', unsafe_allow_html=True)
    else:
        st.warning("Please type a question first.")