# PDF RAG Chatbot

Ask questions about any PDF document using AI.

## Features

- PDF text extraction
- Semantic search with embeddings
- Groq LLM for intelligent answers
- Flask REST API

## Tech Stack

Python, Flask, Groq, SentenceTransformers, PyPDF2, NumPy

## Setup

1. Clone the repo
2. pip install -r requirements.txt
3. Create .env file with GROQ_API_KEY=your_key
4. python app.py

## API Usage

POST /ask
{"question": "What is this document about?"}

## Deployment Note

Requires 1GB+ RAM due to sentence-transformers model size.
Tested locally and on Render (exceeds 512MB free tier).
Architecture uses precomputed FAISS index for optimized retrieval.
