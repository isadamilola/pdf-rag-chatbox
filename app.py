import os
from flask import Flask, request, jsonify
from rag import answer_question, chunks, index

app = Flask(__name__)


@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()

    if not data or not data.get('question'):
        return jsonify({
            'error': 'Please provide a question.'
        }), 400

    question = data['question']
    result = answer_question(question, chunks, index)

    return jsonify({
        'answer': result
    })


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run('0.0.0.0', port=port, debug=False)