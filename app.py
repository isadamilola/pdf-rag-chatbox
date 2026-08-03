from flask import Flask, request, jsonify
from rag import answer_question, extract_text, split_chunks, embed_chunks

app = Flask(__name__)

# I guess there should be some method to extract the file, oh, I remember
text = extract_text('Keytruda_1L-metastatisk NSCLC_ kombinasjonsbehandling_rapport 2 av 2_2020.pdf')
chunks = split_chunks(text)
chunks_embed = embed_chunks(chunks)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question')
    result = answer_question(question, chunks, chunks_embed)

    return jsonify({
        'answer': result
    })

if __name__ == '__main__':
    app.run(debug=True)