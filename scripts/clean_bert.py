import re

def clean_for_bert(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text