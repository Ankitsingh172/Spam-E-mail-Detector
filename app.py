# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1', usecols=[0, 1], names=['label', 'text'], skiprows=1)
df.dropna(inplace=True)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Train the model
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=0)
vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(X_train)
model = LogisticRegression()
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    emails = request.form.getlist('emails')  # Get multiple emails from form
    transformed_emails = vectorizer.transform(emails)
    predictions = model.predict(transformed_emails)
    labels = ['Spam' if pred == 1 else 'Ham' for pred in predictions]

    results = list(zip(emails, labels))
    
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
