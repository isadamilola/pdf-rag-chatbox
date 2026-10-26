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

## Evaluation (Ragas)

6 questions on a public English PDF, with reference answers I checked by hand.
Judge LLM: gpt-oss-20b via Groq. Embeddings: all-MiniLM-L6-v2.

| Config                     | Correct | Faithfulness | Context precision |
| -------------------------- | ------- | ------------ | ----------------- |
| 500 words, top 3           | 5/6     | 0.90         | 0.77              |
| 150 words, top 3 (shipped) | 5/6     | 1.00         | 0.89              |
| 150 words, top 5           | 5/6     | 0.97         | 0.83              |

Findings:

- A chunking bug (words joined without spaces) dropped correct answers to 3/6. Evaluation caught it.
- One question always fails because the question wording matches a heading chunk better than the list chunk.
- With top-5 that failure became a wrong answer that still scored faithfulness 1.0, so I kept top-3, which refuses instead.
- Faithfulness and context precision never compare the answer to the reference. I checked correctness by hand.

Limitations: 6 questions, so differences of ~0.1 are noise. Answer relevancy was too noisy to report. Prompt changed after the first run.
