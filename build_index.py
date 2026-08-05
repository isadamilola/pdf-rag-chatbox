import PyPDF2
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

def extract_text(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def split_chunks(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

model = SentenceTransformer('all-MiniLM-L6-v2')
text = extract_text('Keytruda_1L-metastatisk NSCLC_ kombinasjonsbehandling_rapport 2 av 2_2020.pdf')
chunks = split_chunks(text)
embeddings = model.encode(chunks)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings.astype(np.float32))

faiss.write_index(index, 'index.faiss')
with open('chunks.pkl', 'wb') as f:
    pickle.dump(chunks, f)

print("Index built successfully")