import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt

# -------------------------------
# Setup UI dan Styling
# -------------------------------
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

# -------------------------------
# Judul & Input
# -------------------------------
st.title("🛡️ Deteksi Hoaks dengan Dua Model AI")
st.write("Bandingkan hasil klasifikasi hoaks dari dua model transformer berbasis BERT dan RoBERTa.")

text_input = st.text_area("📄 Masukkan teks berita atau pernyataan:", 
    "COVID-19 vaccines are a government tool to control the population via microchips.")

if not text_input.strip():
    st.warning("⚠️ Masukkan teks terlebih dahulu.")
    st.stop()

# -------------------------------
# Load Dua Model (Cached)
# -------------------------------
@st.cache_resource
def load_models():
    pipe_bert = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")
    pipe_roberta = pipeline("text-classification", model="winterForestStump/Roberta-fake-news-detector")
    return pipe_bert, pipe_roberta

pipe1, pipe2 = load_models()

# -------------------------------
# Prediksi
# -------------------------------
if st.button("🔍 Deteksi Hoaks"):
    res1 = pipe1(text_input)[0]
    res2 = pipe2(text_input)[0]

    st.subheader("📊 Hasil Prediksi")

    for title, res in zip(["🔹 BERT-Tiny", "🔹 RoBERTa"], [res1, res2]):
        st.markdown(f"### {title}")
        st.markdown(f"""
        <div class="result-box">
            <b>Label:</b> {res['label']}<br>
            <b>Confidence:</b> {res['score']:.2f}
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------
    # Visualisasi Confidence
    # -------------------------------
    st.markdown("### 📈 Confidence Score")

    labels = ["FAKE", "REAL"]
    conf1 = [res1['score'] if res1['label'] == l else 1 - res1['score'] for l in labels]
    conf2 = [res2['score'] if res2['label'] == l else 1 - res2['score'] for l in labels]

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

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Model dari 🤗 Hugging Face | Dibuat oleh Rio Anas | UI by Streamlit")
