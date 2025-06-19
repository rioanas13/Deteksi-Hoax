import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

# ----------------------------------------
# Setup UI dan Styling
# ----------------------------------------
st.set_page_config(page_title="Deteksi Hoaks AI", layout="centered")
st.markdown("""
<style>
.result-box {
    background-color: #f9f9f9;
    padding: 1em;
    border-radius: 10px;
    border: 1px solid #ddd;
    margin-bottom: 1em;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    padding: 0.5em 1em;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Deteksi Hoaks dengan Dua Model AI")
st.write("Perbandingan klasifikasi hoaks dari model **BERT-Tiny** dan **RoBERTa** (winterForestStump).")

text_input = st.text_area("📄 Masukkan teks berita atau pernyataan:",
    "The government uses 5G towers to control human minds.")

if not text_input.strip():
    st.warning("⚠️ Masukkan teks terlebih dahulu.")
    st.stop()

# ----------------------------------------
# Load Dua Model (Cached)
# ----------------------------------------
@st.cache_resource
def load_models():
    pipe_bert = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")

    tokenizer = AutoTokenizer.from_pretrained("winterForestStump/Roberta-fake-news-detector")
    model = AutoModelForSequenceClassification.from_pretrained("winterForestStump/Roberta-fake-news-detector")

    return pipe_bert, tokenizer, model

pipe1, roberta_tokenizer, roberta_model = load_models()

# ----------------------------------------
# Prediksi
# ----------------------------------------
if st.button("🔍 Deteksi Hoaks"):
    res1 = pipe1(text_input)[0]

    # RoBERTa manual inference
    inputs = roberta_tokenizer(text_input, return_tensors="pt")
    outputs = roberta_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    pred_class = torch.argmax(probs).item()
    conf_score = probs[0][pred_class].item()

    label_map = {0: "FAKE", 1: "REAL"}
    label2 = label_map[pred_class]

    # Gabung hasil
    st.subheader("📊 Hasil Prediksi")
    for title, label, score in zip(["🔹 BERT-Tiny", "🔹 RoBERTa"],
                                   [res1['label'], label2],
                                   [res1['score'], conf_score]):
        st.markdown(f"### {title}")
        st.markdown(f"""
        <div class="result-box">
            <b>Label:</b> {label}<br>
            <b>Confidence:</b> {score:.2f}
        </div>
        """, unsafe_allow_html=True)

    # Visualisasi
    st.markdown("### 📈 Confidence Score")
    labels = ["FAKE", "REAL"]
    conf1 = [res1['score'] if res1['label'] == l else 1 - res1['score'] for l in labels]
    conf2 = [conf_score if label2 == l else 1 - conf_score for l in labels]

    fig, ax = plt.subplots()
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], conf1, width=0.4, label="BERT-Tiny", color="#4CAF50")
    ax.bar([i + 0.2 for i in x], conf2, width=0.4, label="RoBERTa", color="#FFC107")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Confidence Score")
    ax.set_title("Perbandingan Confidence Dua Model")
    ax.legend()
    st.pyplot(fig)

# ----------------------------------------
# Footer
# ----------------------------------------
st.markdown("---")
st.caption("Model: BERT-Tiny & RoBERTa | Dibuat oleh Rio Anas | UI by Streamlit")
