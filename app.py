from fastapi import FastAPI
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = FastAPI()

# Load the model and Tfidf 
model = pickle.load(open('toxicity_model.pkt', 'rb'))
tfidf = pickle.load(open('tf_idf.pkt', 'rb'))

@app.post("/predict")
async def predict(text: str):
    text_tfidf = tfidf.transform([text]).toarray()
    prediction = model.predict(text_tfidf)
    class_name = "Toxic" if prediction[0] == 1 else "Non-Toxic"
    return {
        "text": text,
        "class": class_name
    }
