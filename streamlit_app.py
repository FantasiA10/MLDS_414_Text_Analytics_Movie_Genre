import streamlit as st
import requests
from collections import Counter

st.set_page_config(page_title="Movie Genre Classifier", layout="centered")

st.markdown("""
    <style>
        .model-box {
            background-color: #f0f2f6;
            padding: 1em;
            border-radius: 0.5em;
            margin-bottom: 1em;
            border-left: 5px solid #0e76a8;
        }
        .final-box {
            background-color: #e6ffe6;
            padding: 1em;
            border-radius: 0.5em;
            margin-top: 1em;
            border-left: 6px solid #28a745;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🎬 Movie Genre Classification")
st.markdown("Enter a movie description to get genre predictions from 4 different models.")

desc = st.text_area("📄 Movie Description", height=220, placeholder="Paste your movie plot here...")

if st.button("🎯 Classify", type="primary"):
    with st.spinner("Fetching predictions..."):
        response = requests.post("http://localhost:8000/predict", json={
            "clean_description": desc,
            "summary": desc
        })

        if response.status_code == 200:
            preds = response.json()
            st.success("Predictions:")

            final_votes = []
            for model_name, result in preds.items():
                pred = result['label']
                prob = result['probs'].get(pred, 0.0)
                final_votes.append(pred)

                st.markdown(f"""
                    <div class="model-box">
                        <strong>🔍 {model_name.replace('_', ' ').title()}</strong><br>
                        Predicted: <span style='color:#d63384'><strong>{pred}</strong></span><br>
                        Probability: <strong>{prob:.2%}</strong>
                    </div>
                """, unsafe_allow_html=True)

            # Final prediction via majority vote
            majority = Counter(final_votes).most_common(1)[0][0]

            st.markdown(f"""
                <div class="final-box">
                    <strong>✅ Final Prediction (Majority Vote):</strong><br>
                    <span style='color:#1a8917; font-size: 1.4em'><strong>{majority}</strong></span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Failed to get prediction. Make sure FastAPI backend is running.")
