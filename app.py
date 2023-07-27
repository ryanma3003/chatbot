# app.py
import numpy as np
import requests
from flask import request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import LinearSVC
from difflib import SequenceMatcher
from flask import Flask
from flask_socketio import SocketIO
from pprint import pprint

app = Flask(__name__)
socketio = SocketIO(app)

chatwoot_url = 'http://192.168.100.43:3000'
chatwoot_bot_token = 'Ux43bL2Bxtq4sQfNCtbmSfrX'

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

# function untuk mengambil pesan
def send_to_bot(sender, message):
    data_rasa = {
        'sender': sender,
        'message': message
    }
    # headers = {"Content-Type": "application/json",
    #            "Accept": "application/json"}

    # r = requests.post(f'{rasa_url}/webhooks/rest/webhook',
    #                   json=data, headers=headers)
    # return r.json()[0]['text']
    
    question_vector = vectorizer.transform([message])
    predicted_class = classifier.predict(question_vector)[0]
    similarity_scores = cosine_similarity(question_vector, faq_vectors)[0]
    best_answer_index = np.argmax(similarity_scores)

    if similarity_scores[best_answer_index] > 0.7:
        message = faq_answers[best_answer_index]
    
        data = {
            'content' : message
        }

        return data
    else:
        similar_questions = []
        for i, q in enumerate(faq_data):
            similarity_ratio = SequenceMatcher(None, message, q).ratio()
            if similarity_ratio > 0.7:  
                similar_questions.append(faq_data[i])

        if similar_questions:
            message = "Apakah yang Anda maksud dengan:\n" + "\n".join(similar_questions)
        else:
            message = "Maaf, saya tidak mengerti pertanyaan Anda."
    
        data = {
            'content' : message
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

# Inisialisasi vektorisasi TfidfVectorizer
vectorizer = TfidfVectorizer()

# Melakukan vektorisasi pada data pertanyaan
faq_vectors = vectorizer.fit_transform(faq_data)

# Melatih model klasifikasi LinearSVC
classifier = LinearSVC()
classifier.fit(faq_vectors, np.arange(len(faq_data)))

# Route for create outgoing and incoming message chatwoot
@app.route("/", methods=['POST'])
def bot():
    data = request.get_json()
    message_type = data['message_type']
    message = data['content_attributes']['submitted_values'][0]['value']
    conversation = data['conversation']['display_id']
    contact = data['sender']['id']
    account = data['account']['id']

    if(message_type == "incoming"):
        bot_response = send_to_bot(contact, message)
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
    app.run(host='0.0.0.0')
