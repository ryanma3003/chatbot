# app.py
import numpy as np
import requests
import torch
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from transformers import BertTokenizer, BertModel
from difflib import SequenceMatcher
from flask import Flask, request
from flask_socketio import SocketIO
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from googletrans import Translator
from pprint import pprint

app = Flask(__name__)
socketio = SocketIO(app)

chatwoot_url = 'http://192.168.100.43:3000'
chatwoot_bot_token = 'Ux43bL2Bxtq4sQfNCtbmSfrX'

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

        # returned data for option, need more logic to meet all the question level
        data = {
            'content': answer,
            'content_type': 'input_select',
            'content_attributes': {
                "items": [
                    {
                        "title": "Apa itu Paten",
                        "value": "apa yang dimaksud paten"
                    },
                    {
                        "title": "Apa itu invensi",
                        "value": "apa yang dimaksud invensi"
                    },
                    {
                        "title": "Apa itu Paten Sederhana",
                        "value": "apa yang dimaksud paten sederhana"
                    }
                ],
            },
        }

        return data
    else:
        # Mencari pertanyaan serupa
        similar_questions = []
        for i, q in enumerate(faq_data):
            similarity_ratio = SequenceMatcher(None, translated_question, q).ratio()
            if similarity_ratio > 0.5:  # Menentukan batas kemiripan yang diinginkan
                similar_questions.append(faq_data[i])

        if similar_questions:
            answer = "Apakah yang Anda maksud dengan:\n" + "\n".join(similar_questions)

            # returned data for similar question, need more logic to escalate the option
            data = {
                'content': answer,
                'content_type': 'input_select',
                'content_attributes': {
                    "items": [
                        {
                            "title": "Apa itu Paten",
                            "value": "apa yang dimaksud paten"
                        },
                        {
                            "title": "Apa itu invensi",
                            "value": "apa yang dimaksud invensi"
                        },
                        {
                            "title": "Apa itu Paten Sederhana",
                            "value": "apa yang dimaksud paten sederhana"
                        }
                    ],
                },
            }
        else:
            answer = "Maaf, saya tidak mengerti pertanyaan Anda."

            # returned data for failure, need more logic for handover to live agent
            data = {
                'content': answer,
                'content_type': 'input_select',
                'content_attributes': {
                    "items": [
                        {
                            "title": "Bertanya dengan live agent",
                            "value": "live agent"
                        }
                    ],
                },
            }

        return data

def greet():
    data = {
        'content': 'Halo selamat datang, bot kami akan membantu Anda.',
        'content_type': 'input_select',
        'content_attributes': {
            "items": [
                {
                    "title": "Apa itu Paten",
                    "value": "apa yang dimaksud paten"
                },
                {
                    "title": "Apa itu invensi",
                    "value": "apa yang dimaksud invensi"
                },
                {
                    "title": "Apa itu Paten Sederhana",
                    "value": "apa yang dimaksud paten sederhana"
                }
            ],
        },
    }

    return data

# function untuk mengirim pesan
def send_to_chatwoot(account, conversation, data_json):
    data = data_json
    url = f"{chatwoot_url}/api/v1/accounts/{account}/conversations/{conversation}/messages"
    headers = {"Content-Type": "application/json",
               "Accept": "application/json",
               "api_access_token": f"{chatwoot_bot_token}"}

    r = requests.post(url,
                      json=data, headers=headers)
    return r.json()

# Route for create outgoing and incoming message chatwoot
@app.route("/", methods=['POST'])
def bot():
    data = request.get_json()
    pprint(data)
    message_type = data['message_type']

    if(message_type == "incoming"):
        message = data['content']
        conversation = data['conversation']['id']
        contact = data['sender']['id']
        account = data['account']['id']

        bot_response = greet()
        create_message = send_to_chatwoot(
            account, conversation, bot_response)
        
    elif(message_type == "outgoing"):
        message = data['content_attributes']['submitted_values'][0]['value']
        conversation = data['conversation']['id']
        contact = data['sender']['id']
        account = data['account']['id']

        bot_response = get_best_answer(message)
        create_message = send_to_chatwoot(
            account, conversation, bot_response)
    return create_message

# Route for webhook event start conversation
@app.route("/webhook", methods=['POST'])
def greeting():
    data = request.get_json()
    message_type = data['message_type']
    message = data['content_attributes']['submitted_values'][0]['value']
    conversation = data['conversation']['display_id']
    contact = data['sender']['id']
    account = data['account']['id']

    if(message_type == "incoming"):
        bot_response = greet()
        create_message = send_to_chatwoot(
            account, conversation, bot_response)
    return create_message

if __name__ == "__main__":
    app.run(host='192.168.100.25')
