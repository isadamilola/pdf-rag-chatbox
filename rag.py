import PyPDF2
from sentence_transformers import SentenceTransformer
from groq import Groq
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

# Step 1: Extract text from PDF
def extract_text(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Step 2: Split into chunks
def split_chunks(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# Test it
text = extract_text('Keytruda_1L-metastatisk NSCLC_ kombinasjonsbehandling_rapport 2 av 2_2020.pdf')
chunks = split_chunks(text)
# print(f"Total chunks: {len(chunks)}")
# print(f"First chunk preview: {chunks[0][:200]}")

# Step 3: Convert chunks to embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_chunks(chunks):
    embeddings = embedding_model.encode(chunks)
    return embeddings

chunk_embeddings = embed_chunks(chunks)
#print(f"Embeddings shape: {chunk_embeddings.shape}")

from sklearn.metrics.pairwise import cosine_similarity

def retrieve_chunks(question, chunks, chunk_embeddings, top_k=3):
    question_embedding = embedding_model.encode([question])
    similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [chunks[i] for i in top_indices]

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def answer_question(question, chunks, chunk_embeddings):
    relevant_chunks = retrieve_chunks(question, chunks, chunk_embeddings)
    context = "\n\n".join(relevant_chunks)
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": f"Answer questions based only on this context:\n\n{context}"},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content

print(answer_question("What is this document about?", chunks, chunk_embeddings))