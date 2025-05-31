# Multimodal Movie Genre Classification

**Northwestern University — Text Analytics**  
**Final Project — Spring 2025**
**Team Member: Jerry Zhu, Mason Ma, Yishi Wang, Yunkai Jin, Seung Jae Lee**

## Overview

This project explores genre classification of movies using two modalities: **plot summaries** (text) and **poster images** (vision). The primary focus is on building and comparing multiple NLP models for genre prediction, while the optional bonus challenge applies transfer learning to classify genres using movie posters.

We target four genres: **Action**, **Comedy**, **Horror**, and **Romance**.

## Project Objectives

- Clean and analyze movie summaries
- Build a summarization tool
- Train multiple genre classifiers using text:
  - Naive Bayes
  - Logistic Regression
  - Random Forest
  - LSTM (TensorFlow)
- Evaluate models using overall and genre-wise accuracy
- Visualize model performance and word importance
- Identify and explain misclassified examples
- Deploy an interactive app for real-time genre prediction
- **(Bonus)**: Classify genre based on poster images using MobileNetV2

---

## Directory Structure

```bash
FINAL_PROJECT/
├── bonus_problem/                # Bonus task: poster-based classifier
│   ├── mobilenet_genre_classifier.h5
│   ├── mobilenet_genre_finetuned.h5
│   ├── mobilenet_v2_weights_tf_dim_ordering_tf_kernels.h5
│   └── datasets/                # Poster image dataset (ignored by git)
│
├── datasets/                    # Text data and cleaned files (ignored by git)
├── models/                      # Saved models and checkpoints
├── scripts/                     # Utility and training scripts
├── classify-by-img.ipynb       # Bonus: Poster classification notebook
├── classify-by-text.ipynb      # Main text classification notebook
├── DistilBERT_genre_training.py
├── DistilBERT_genre_inference.py
├── LR and LSTM.ipynb
├── nb_rf.ipynb
├── streamlit_app.py            # Interactive genre prediction app
├── train_all_models.py         # End-to-end training pipeline
├── requirements.txt
├── README.md
└── Project Discussion.html     # Slides or written presentation
```


## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/FantasiA10/MLDS_414_Text_Analytics_Movie_Genre.git
cd MLDS_414_Text_Analytics_Movie_Genre
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Web App

```bash
streamlit run streamlit_app.py
```

---

## Model Highlights

* **Naive Bayes**: Fast baseline using bag-of-words features
* **Logistic Regression**: Interpretable linear classifier
* **Random Forest**: Captures non-linearities
* **LSTM (TensorFlow)**: Sequence-aware deep learning
* **DistilBERT**: Transformer-based classifier (optional)
* **MobileNetV2 (Bonus)**: CNN-based model for poster classification

Each model was evaluated using cross-validation and accuracy by genre. LSTM training curves and word clouds provide insight into model learning and interpretability.

---

## Bonus: Poster-Based Genre Classification

Under the `/bonus_problem/` directory, we trained a **MobileNetV2-based model** using poster images. This model was fine-tuned for our four genres and achieved promising accuracy, showcasing the power of visual cues in genre prediction.

---

## Deliverables

* [x] Code and models in this GitHub repository
* [x] Slide deck (`Text Analytics Movie Genre Classification.pdf`)
* [x] Final presentation video (see course submission)

---

---

## Notes

* Large data files (CSV/images) in `datasets/` and `bonus_problem/datasets/` are `.gitignore`d.
* Contact the team if you need access to the full dataset for reproduction.
