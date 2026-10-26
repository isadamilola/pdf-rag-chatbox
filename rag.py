import faiss
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
def split_chunks(text, chunk_size=150, overlap=30):
    words = text.split()
    step = chunk_size - overlap
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), step)]

# Test it
text = extract_text('Small patient groups_final.pdf')
chunks = split_chunks(text)
# print(f"Total chunks: {len(chunks)}")
# print(f"First chunk preview: {chunks[0][:200]}")

# Step 3: Convert chunks to embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def build_faiss_index(chunks):
    embeddings = embed_chunks(chunks).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings

def embed_chunks(chunks):
    embeddings = embedding_model.encode(chunks)
    return embeddings

index, chunk_embeddings = build_faiss_index(chunks)
#print(f"Embeddings shape: {chunk_embeddings.shape}")

def retrieve_chunks(question, chunks, index, top_k=3):
    question_embedding = embedding_model.encode([question]).astype("float32")

    distances, indices = index.search(question_embedding, top_k)

    return [chunks[i] for i in indices[0]]

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def answer_question(question, chunks, index):
    relevant_chunks = retrieve_chunks(question, chunks, index)
    context = "\n\n".join(relevant_chunks)
    
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
    "role": "system",
    "content": (
        "Answer the user's question using only the provided context. "
        "Be concise and answer directly, usually in 1-3 sentences. "
        "Do not add unrelated details, assumptions, or information "
        "that is not supported by the context. If the answer is not "
        "available in the context, say so.\n\n"
        f"Context:\n{context}"
    ),
},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print(answer_question("What is this document about?", chunks, index))