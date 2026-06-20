# Clinical NLP RAG System

A question-answering system for medical research. You ask a clinical question, and it pulls relevant information from a database of 211,000 PubMed papers and writes back a grounded answer. It runs a language model locally, so nothing gets sent to an external service, which matters when you're working with sensitive medical content.

I built this to get hands-on with the kind of RAG pipelines that come up a lot in NLP and GenAI roles, and to build on my background in medical AI.

## What it does

You type a medical question. The system finds the most relevant research papers using semantic search, then passes those papers to a local language model that answers using only what's in them. If the papers don't actually cover the question, it says so instead of making something up.

## Features

- Semantic search across 211,269 medical research papers
- FAISS vector database for fast retrieval
- Answers grounded in retrieved papers, not the model's own memory
- Local LLM (Llama 3.2 via Ollama), so data stays on the machine
- FastAPI backend with search and ask endpoints
- Streamlit web interface with source papers shown alongside each answer

## Tech stack

Python, FastAPI, Streamlit, Sentence Transformers, FAISS, Ollama (Llama 3.2), Pandas, NumPy.

## How it works

The medical documents are cleaned and turned into 384-dimensional vectors using Sentence Transformers, then stored in a FAISS index. When you ask a question, it gets converted into a vector and matched against the index to find the closest papers. Those papers and your question go to the local model, which writes an answer based only on them. The answer comes back along with the source papers so you can check it against the originals.

## Project structure

clinical-nlp-rag/
├── notebooks/
│   ├── 1_data_cleaning.ipynb
│   ├── embeddings.ipynb
│   └── vector_database.ipynb
├── main.py            # FastAPI backend
├── app_ui.py          # Streamlit frontend
├── data/processed/    # cleaned data, embeddings, FAISS index
├── requirements.txt
└── README.md

## Running it

Start the backend:

uvicorn main:app

Then start the interface in a second terminal:

streamlit run app_ui.py

## Results

The system indexes 211,269 clinical text chunks as 384-dimensional vectors and retrieves matches in under a second. Answers are grounded in real papers and come with their sources attached. The whole pipeline runs locally.

## Future work

A few things I'd like to add next:

- Hybrid search. Right now retrieval is purely semantic (FAISS). Medical text has a lot of exact terms, drug names, and abbreviations where keyword matching does better, so I want to combine FAISS with BM25 keyword search and merge the scores.
- Confidence scores. The model already admits when papers don't cover a question, but I'd like to attach a numeric confidence to each answer so a user can quickly see how reliable it is.
- Show retrieval quality. Surface the similarity scores in the interface so it's clear how strong the matches were.
- Bigger and cleaner data. Handle the label imbalance in the dataset and test the system on a larger medical corpus.
- Evaluation. Set up a proper evaluation of answer quality against a set of known question-answer pairs, rather than judging by eye.

## Author

Anjali Roy Rathi
