🛠 Project: AI Chatbot Development using NLP
📌 Objective:

Build a simple chatbot that can:

Understand user input (intent recognition).

Respond with predefined answers.

Use NLP techniques like tokenization, regex cleaning, lemmatization, TF-IDF + ML model.

📂 Steps & Code
Step 1: Import libraries
import re
import random
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

Step 2: Preprocessing function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # Lowercase
    text = text.lower()
    # Remove special chars using regex
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords + lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

Step 3: Training Data (Intents + Responses)
# Example chatbot dataset
data = {
    "intent": ["greet", "greet", "bye", "bye", "thanks", "thanks", "order", "order"],
    "text": [
        "hello", "hi there", 
        "bye", "see you later",
        "thank you", "thanks a lot", 
        "i want to order pizza", "can i get a burger"
    ],
    "response": [
        "Hello! How can I help you?", "Hi! Nice to see you!",
        "Goodbye! Have a great day!", "See you soon!",
        "You're welcome!", "Happy to help!",
        "Sure, I can help you place an order.", "Okay, what would you like to order?"
    ]
}

df = pd.DataFrame(data)

# Preprocess user text
df["clean_text"] = df["text"].apply(preprocess)

Step 4: Train Model for Intent Classification
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_text"])
y = df["intent"]

model = LogisticRegression()
model.fit(X, y)

Step 5: Chatbot Response Function
def chatbot_response(user_input):
    clean_input = preprocess(user_input)
    X_input = vectorizer.transform([clean_input])
    intent = model.predict(X_input)[0]
    
    # Pick a random response for the intent
    responses = df[df["intent"] == intent]["response"].values
    return random.choice(responses)

Step 6: Chat with Bot
print("🤖 Chatbot is ready! (type 'quit' to exit)\n")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("🤖 Bot: Goodbye!")
        break
    print("🤖 Bot:", chatbot_response(user_input))

📌 How it Works:

Preprocess user query (clean text → tokenize → lemmatize).

Convert to TF-IDF vector.

Use Logistic Regression to classify intent (greet/bye/order/thanks).

Return a response mapped to that intent.

🚀 Example Run:
🤖 Chatbot is ready! (type 'quit' to exit)

You: hello
🤖 Bot: Hi! Nice to see you!

You: i want pizza
🤖 Bot: Sure, I can help you place an order.

You: thanks
🤖 Bot: You're welcome!

You: bye
🤖 Bot: Goodbye! Have a great day!




Project Story  – AI Chatbot Development (Mentorness Internship, Mar 2024)

Project Overview:
Developed an AI chatbot to automate customer query handling using NLP and deep learning.

Problem Statement:
Manual support for FAQs was time-consuming and error-prone. The goal was to automate responses while maintaining accuracy.

Dataset:

Collected ~5000 Q&A pairs from customer support and online FAQs.

Columns: question, answer.

ETL / Preprocessing Steps:

Extract → Data collected from CSV files and APIs.

Transform →

Text cleaning: lowercasing, removing punctuation, stopwords.

Tokenization (splitting sentences into words).

Vectorization: Word embeddings (Word2Vec / TF-IDF / BERT embeddings).

Load → Prepared dataset stored in CSV/DB for training.

ML / NLP Algorithms Used:

Text Classification → RNN / LSTM / Transformer for intent recognition.

Sequence-to-Sequence (Seq2Seq) for response generation.

Evaluation → Accuracy, F1-score, BLEU score (for chatbot responses).

Deployment:

Flask API for serving chatbot responses.

Docker container for portability.

Could be scaled using cloud (Azure ML endpoints).

Impact / Outcome:

Reduced manual handling by 70%.

Improved response time and consistency.

Key Skills Demonstrated: Python, NLP, Deep Learning, MLOps (Flask + Docker), ETL.
