# frontend/streamlit_app.py
import streamlit as st
import requests

st.set_page_config(page_title="Movie Genre Classifier", layout="centered")
st.title("🎬 IMDB Movie Genre Classifier")

description = st.text_area("Enter a movie description:", height=200)

if st.button("Predict Genre"):
    if not description.strip():
        st.warning("Please enter a description.")
    else:
        with st.spinner("Classifying..."):
            try:
                response = requests.post(
                    "http://localhost:8000/predict",
                    json={"description": description}
                )
                result = response.json()
                st.success(f"Predicted Genre: **{result['genre'].capitalize()}**")
            except Exception as e:
                st.error(f"Error: {str(e)}")
