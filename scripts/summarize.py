import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def summarize_text(text, num_sentences=2):
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)

    svd = TruncatedSVD(n_components=1, random_state=42)
    svd_matrix = svd.fit_transform(X)

    top_sentence_idx = np.argsort(svd_matrix[:, 0])[::-1][:num_sentences]
    top_sentence_idx.sort()

    summary = ' '.join([sentences[i] for i in top_sentence_idx])
    return summary