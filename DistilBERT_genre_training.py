import os
import json
import pickle

from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd

from scripts.summarize import summarize_text
from scripts.clean_bert import clean_for_bert

# Load dataset
df = pd.read_csv('./datasets/imdb/IMDB_four_genre_larger_plot_description.csv')
df = df[['description', 'genre']].dropna()
print("Dataset Loaded")

df['clean_description'] = df['description'].astype(str).apply(clean_for_bert)
print("Data Cleaned")

df['summary'] = df['clean_description'].apply(lambda x: summarize_text(x, num_sentences=5))
print("Summary Created")

# Encode target labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['genre'])  # Save this encoder if needed for inference

# Split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Convert to HuggingFace Datasets
train_ds = Dataset.from_pandas(train_df[['summary', 'label']])
val_ds = Dataset.from_pandas(val_df[['summary', 'label']])

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['summary'], truncation=True, padding=True)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# Load model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=500, 
    save_total_limit=1
)

# Metric
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Evaluate
results = trainer.evaluate()
print("Final Evaluation Results:", results)

# Save results
os.makedirs('./bert_genre_model', exist_ok=True)
with open('./bert_genre_model/eval_results.json', 'w') as f:
    json.dump(results, f, indent=4)
with open('./bert_genre_model/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Save model and tokenizer
model.save_pretrained('./bert_genre_model')
tokenizer.save_pretrained('./bert_genre_model')

