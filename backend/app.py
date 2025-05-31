from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = FastAPI()

# Load all models and preprocessors
le = pickle.load(open('models/label_encoder.pkl', 'rb'))
nb_model = pickle.load(open('models/nb_model.pkl', 'rb'))
nb_vectorizer = pickle.load(open('models/nb_vectorizer.pkl', 'rb'))
rf_model = pickle.load(open('models/rf_model.pkl', 'rb'))
rf_vectorizer = pickle.load(open('models/rf_vectorizer.pkl', 'rb'))
lr_model = pickle.load(open('models/lr_model.pkl', 'rb'))
lr_vectorizer = pickle.load(open('models/lr_vectorizer.pkl', 'rb'))
lstm_model = tf.keras.models.load_model('models/imdb_lstm_model.h5')
lstm_tokenizer = pickle.load(open('models/lstm_tokenizer.pkl', 'rb'))

class MovieInput(BaseModel):
    clean_description: str
    summary: str

@app.post("/predict")
def predict_genre(inp: MovieInput):
    results = {}

    # Naive Bayes
    X_nb = nb_vectorizer.transform([inp.clean_description])
    nb_probs = nb_model.predict_proba(X_nb)[0]
    nb_pred = nb_model.classes_[np.argmax(nb_probs)]
    results["naive_bayes"] = {
        "label": nb_pred,
        "probs": dict(zip(nb_model.classes_, nb_probs.tolist()))
    }

    # Random Forest
    X_rf = rf_vectorizer.transform([inp.summary + " " + inp.clean_description])
    rf_probs = rf_model.predict_proba(X_rf)[0]
    rf_pred = rf_model.classes_[np.argmax(rf_probs)]
    results["random_forest"] = {
        "label": rf_pred,
        "probs": dict(zip(rf_model.classes_, rf_probs.tolist()))
    }

    # Logistic Regression
    X_lr = lr_vectorizer.transform([inp.summary])
    lr_probs = lr_model.predict_proba(X_lr)[0]
    lr_pred = le.inverse_transform([np.argmax(lr_probs)])[0]
    genre_labels = le.classes_.tolist()
    results["logistic_regression"] = {
        "label": lr_pred,
        "probs": dict(zip(genre_labels, lr_probs.tolist()))
    }

    # LSTM
    X_lstm = lstm_tokenizer.texts_to_sequences([inp.summary])
    X_lstm = tf.keras.preprocessing.sequence.pad_sequences(X_lstm, maxlen=300)
    lstm_probs = lstm_model.predict(X_lstm)[0]
    lstm_pred = le.inverse_transform([np.argmax(lstm_probs)])[0]
    results["lstm"] = {
        "label": lstm_pred,
        "probs": dict(zip(genre_labels, lstm_probs.tolist()))
    }

    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)