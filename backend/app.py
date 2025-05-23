# backend/app.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from scripts.clean import clean_text
from scripts.summarize import summarize_text

class MovieInput(BaseModel):
    description: str

# Load model, tokenizer, label encoder
model_path = "./backend/bert_genre_model"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

with open(f"{model_path}/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

app = FastAPI()

@app.post("/predict")
def predict_genre(input: MovieInput):

    if not input.description.strip():
        raise HTTPException(status_code=400, detail="Empty description.")

    cleaned = clean_text(input.description)
    summary = summarize_text(cleaned)

    inputs = tokenizer(summary, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    genre = le.inverse_transform([prediction])[0]
    return {"summary": summary, "genre": genre}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
