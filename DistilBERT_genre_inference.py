# genre_inference.py
import torch
import pickle
import pandas as pd
import argparse

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from scripts.summarize import summarize_text
from scripts.clean import clean_text

# Load saved model and tokenizer
model_path = './bert_genre_model'
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()  # set to eval mode

with open('./bert_genre_model/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

def predict_genre(texts):
    texts = clean_text(texts)
    texts = summarize_text(texts)
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1)
    return le.inverse_transform(preds.numpy())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, help='Movie description to classify')
    args = parser.parse_args()

    if args.text:
        result = predict_genre([args.text])
        print(f"Predicted Genre: {result[0]}")
    else:
        print("Please provide a --text argument with the movie description.")
