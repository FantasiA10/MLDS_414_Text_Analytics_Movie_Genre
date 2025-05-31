import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

print(tf.config.list_physical_devices('GPU'))

# Load Data
df = pd.read_csv('datasets/imdb/imdb_cleaned.csv')
X_nb = df['clean_description']
X_rf = df['summary'] + " " + df['clean_description']
X_lr = df['summary']
X_lstm = df['summary']
y = df['genre']

# Split once for all models
X_train_idx, X_test_idx, y_train, y_test = train_test_split(
    df.index, y, stratify=y, test_size=0.2, random_state=42
)
# Naive Bayes
X_nb_train, X_nb_test = X_nb.loc[X_train_idx], X_nb.loc[X_test_idx]
# Random Forest
X_rf_train, X_rf_test = X_rf.loc[X_train_idx], X_rf.loc[X_test_idx]
# Logistic Regression
X_lr_train, X_lr_test = X_lr.loc[X_train_idx], X_lr.loc[X_test_idx]
# LSTM
X_lstm_train, X_lstm_test = X_lstm.loc[X_train_idx], X_lstm.loc[X_test_idx]
y_train, y_test = y.loc[X_train_idx], y.loc[X_test_idx]

# Label encode
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

os.makedirs("models", exist_ok=True)
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Naive Bayes
vectorizer_nb = TfidfVectorizer(max_features=5000)
X_nb_train_vec = vectorizer_nb.fit_transform(X_nb_train)
X_nb_test_vec = vectorizer_nb.transform(X_nb_test)
nb_model = MultinomialNB().fit(X_nb_train_vec, y_train)
pickle.dump(nb_model, open('models/nb_model.pkl', 'wb'))
pickle.dump(vectorizer_nb, open('models/nb_vectorizer.pkl', 'wb'))

# Random Forest
vectorizer_rf = TfidfVectorizer(max_features=4000)
X_rf_train_vec = vectorizer_rf.fit_transform(X_rf_train)
X_rf_test_vec = vectorizer_rf.transform(X_rf_test)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_rf_train_vec, y_train)
pickle.dump(rf_model, open('models/rf_model.pkl', 'wb'))
pickle.dump(vectorizer_rf, open('models/rf_vectorizer.pkl', 'wb'))

# Logistic Regression
vectorizer_lr = TfidfVectorizer(max_features=5000, stop_words='english')
X_lr_train_vec = vectorizer_lr.fit_transform(X_lr_train)
X_lr_test_vec = vectorizer_lr.transform(X_lr_test)
lr_model = LogisticRegression(max_iter=1000).fit(X_lr_train_vec, y_train_enc)
pickle.dump(lr_model, open('models/lr_model.pkl', 'wb'))
pickle.dump(vectorizer_lr, open('models/lr_vectorizer.pkl', 'wb'))

# LSTM
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_lstm_train)
X_lstm_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_lstm_train), maxlen=300)
X_lstm_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_lstm_test), maxlen=300)
y_train_cat = to_categorical(y_train_enc)
y_test_cat = to_categorical(y_test_enc)
lstm_model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=300),
    LSTM(64),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
])
lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.fit(X_lstm_train_seq, y_train_cat, epochs=10, batch_size=64, validation_split=0.1)
lstm_model.save('models/imdb_lstm_model.h5')
pickle.dump(tokenizer, open('models/lstm_tokenizer.pkl', 'wb'))

print("All models saved to ./models/")
