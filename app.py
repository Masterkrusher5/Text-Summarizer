from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import pipeline
import fitz
import os

app = Flask(__name__)
CORS(app)

summarizer = pipeline("summarization", model="MK-5/t5-small-Abstractive-Summarizer")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    input_data = request.json
    text = input_data.get("text", "")
    max_length = input_data.get("max_length", 50)

    try:
        max_length = int(max_length)
    except ValueError:
        return jsonify({"error": "max_length must be an integer."}), 400

    if not text.strip():
        return jsonify({"error": "Input text cannot be empty or consist only of symbols."}), 400

    try:

        summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)[0]["summary_text"]
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": f"An error occurred during summarization: {str(e)}"}), 500

@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        if file.filename.endswith('.pdf'):
            pdf_document = fitz.open(stream=file.read(), filetype="pdf")
            text = ""
            for page in pdf_document:
                text += page.get_text()
            pdf_document.close()
            return jsonify({"text": text})
        elif file.filename.endswith('.txt'):
            text = file.read().decode('utf-8')
            return jsonify({"text": text})

        else:
            return jsonify({"error": "Unsupported file type. Please upload a .txt or .pdf file."}), 400

    except Exception as e:
        return jsonify({"error": f"Failed to extract text from file: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
