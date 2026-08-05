import os
import faiss
import pickle
from flask import Flask, request, jsonify
from rag import answer_question

app = Flask(__name__)

# Load prebuilt index - no model loading here
index = faiss.read_index('index.faiss')
with open('chunks.pkl', 'rb') as f:
    chunks = pickle.load(f)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question')
    result = answer_question(question, chunks, index)

    return jsonify({
        'answer': result
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run('0.0.0.0', port=port, debug=False)