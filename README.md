# Clinical NLP RAG System

## About

This project explores how Retrieval-Augmented Generation (RAG) can be applied to clinical text data.

The system processes medical documents, generates embeddings using Sentence Transformers, stores them in a FAISS vector database, and retrieves relevant information based on user queries.

## Features

- Data cleaning and preprocessing
- Document chunking
- Sentence Transformer embeddings
- FAISS vector database
- Semantic search

## Tech Stack

- Python
- Pandas
- NumPy
- Sentence Transformers
- FAISS
- Jupyter Notebook

## Project Structure

text clinical-nlp-rag/ │ ├── 1_data_cleaning.ipynb ├── Chunking.ipynb ├── embeddings.ipynb ├── requirements.txt └── README.md 

## Results

- Generated embeddings for 211,269 clinical text chunks
- Embedding dimension: 384
- Stored vectors in FAISS vector database
- Ready for similarity search

## Future Work

- Implement document retrieval
- Integrate LLM
- Build complete RAG pipeline

## Author

Anjali Roy Rathi