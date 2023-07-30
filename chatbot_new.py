# app.py
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from difflib import SequenceMatcher
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from googletrans import Translator

app = Flask(__name__)
socketio = SocketIO(app)

# Function to read data from the text file with a specific delimiter
def read_data_from_file(file_path, delimiter="|"):
    with open(file_path, "r", encoding="utf-8") as file:
        data_lines = file.readlines()
    faq_data, faq_answers = [], []
    for line in data_lines:
        question, answer = line.strip().split(delimiter)
        faq_data.append(question.strip())
        faq_answers.append(answer.strip())
    return faq_data, faq_answers

# File path to the text file containing FAQ data
faq_file_path = "data/faq_data.txt"

# Data Frequently Asked Questions and answers read from the file
faq_data, faq_answers = read_data_from_file(faq_file_path)

# Struktur pertanyaan dan jawaban dengan opsi bertingkat hingga 5 level
questions = {
    "question": "Apa yang ingin Anda ketahui?",
    "options": [
        {
            "id": 1,
            "text": "Level 1: Apa yang dimaksud Paten?",
            "next_question": {
                "question": "Paten adalah hak eksklusif yang diberikan oleh negara kepada inventor atas hasil invensinya di bidang teknologi untuk jangka waktu tertentu melaksanakan sendiri invensi tersebut atau memberikan persetujuan kepada pihak lain untuk melaksanakannya",
                "options": [
                    {
                        "id": 11,
                        "text": "Apa yang dimaksud Paten sederhana?",
                        "next_question": {
                            "question": "Setiap invensi berupa produk atau alat yang baru dan mempunyai nilai kegunaan praktis disebabkan karena bentuk, konfigurasi, konstruksi atau komponennya dapat memperoleh perlindungan hukum dalam bentuk paten sederhana",
                            "options": [
                                {
                                    "id": 111,
                                    "text": "Level 3: Topik 1.1.1",
                                    "next_question": {
                                        "question": "Pertanyaan untuk Level 3: Topik 1.1.1",
                                        "options": [
                                            {
                                                "id": 1111,
                                                "text": "Level 4: Topik 1.1.1.1",
                                                "next_question": {
                                                    "question": "Pertanyaan untuk Level 4: Topik 1.1.1.1",
                                                    "options": [
                                                        {
                                                            "id": 11111,
                                                            "text": "Level 5: Topik 1.1.1.1.1",
                                                            "next_question": None
                                                        }
                                                    ]
                                                }
                                            }
                                        ]
                                    }
                                },
                                {
                                    "id": 112,
                                    "text": "Level 3: Topik 1.1.2",
                                    "next_question": None
                                }
                            ]
                        },
                        "id": 22,
                        "text": "Apa yang dimaksud Invensi?",
                        "next_question": {
                            "question": "Invensi adalah ide inventor yang dituangkan ke dalam suatu kegiatan pemecahan masalah yang spesifik di bidang teknologi. dapat berupa produk atau proses atau penyempurnaan dan pengembangan produk atau proses",
                            "options": [
                                {
                                    "id": 222,
                                    "text": "Apa saja yang harus dicantumkan dalam surat Pernyataan Kepemilikan Invensi paten?",
                                    "next_question": None
                                },
                            ]
                        }
                    }
                ]
            }
        },
        {
            "id": 2,
            "text": "Level 1: Topik 2",
            "next_question": None
        }
    ]
}

# Inisialisasi tokenizer dan model BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

# Fungsi untuk mendapatkan vektor representasi pertanyaan dari model BERT
def get_bert_vector(question):
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Menghasilkan vektor representasi dari FAQ data menggunakan model BERT
faq_vectors = [get_bert_vector(faq) for faq in faq_data]

# Create a Translator object for translation
translator = Translator()

# Fungsi untuk mendapatkan jawaban terbaik atau opsi pilihan
def get_best_answer(question):
    # Detect the language of the input question
    lang = detect(question)

    # If the language is English, translate it to Indonesian
    if lang == "en":
        translated_question = translator.translate(question, src="en", dest="id").text
    else:
        translated_question = question

    # Vektorisasi pertanyaan menggunakan model BERT
    question_vector = get_bert_vector(translated_question)

    # Menghitung skor cosine similarity
    similarity_scores = [cosine_similarity(question_vector.reshape(1, -1), faq_vector.reshape(1, -1))[0][0] for faq_vector in faq_vectors]

    # Mengambil jawaban dengan skor tertinggi
    best_answer_index = np.argmax(similarity_scores)

    if similarity_scores[best_answer_index] > 0.5:
        answer = faq_answers[best_answer_index]
        # If the original question was in English, translate the answer back to English
        if lang == "en":
            answer = translator.translate(answer, src="id", dest="en").text
        return answer
    else:
        # Mencari pertanyaan serupa
        similar_questions = []
        for i, q in enumerate(faq_data):
            similarity_ratio = SequenceMatcher(None, translated_question, q).ratio()
            if similarity_ratio > 0.5:  # Menentukan batas kemiripan yang diinginkan
                similar_questions.append(faq_data[i])

        if similar_questions:
            return "Apakah yang Anda maksud dengan:\n" + "\n".join(similar_questions)
        else:
            return "Maaf, saya tidak mengerti pertanyaan Anda."

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("user_question")
def handle_user_question(data):
    question = data["question"]
    option = data["option"]

    if question:
        bot_answer = get_best_answer(question)
        socketio.emit("bot_answer", {"answer": bot_answer, "option": option})

@app.route('/api/question', methods=['GET'])
def get_question():
    return jsonify(questions)

@app.route("/chatbot")
def chatbot():
    return render_template("index.html")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)