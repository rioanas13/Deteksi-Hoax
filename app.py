import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt

# -------------------------------
# Setup halaman & gaya
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
# Judul & input teks
# -------------------------------
st.title("🛡️ Deteksi Hoaks dengan Dua Model AI")
st.write("Masukkan teks berita atau opini, lalu lihat prediksi dari dua model NLP: **BERT-Tiny** dan **RoBERTa**.")

text_input = st.text_area("📄 Masukkan teks:", 
    "The COVID-19 vaccine contains microchips to track citizens.")

if not text_input.strip():
    st.warning("⚠️ Harap masukkan teks terlebih dahulu.")
    st.stop()

# -------------------------------
# Load model dengan caching
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

    # Konversi label dari RoBERTa (0 → FAKE, 1 → REAL)
    label_map = {
        0: "FAKE", 1: "REAL",
        "0": "FAKE", "1": "REAL",
        "LABEL_0": "FAKE", "LABEL_1": "REAL"
    }
    res2['label'] = label_map.get(res2['label'], str(res2['label']))

    st.subheader("📊 Hasil Prediksi")

    for title, res in zip(["🔹 BERT-Tiny", "🔹 RoBERTa"], [res1, res2]):
        st.markdown(f"### {title}")
        st.markdown(f"""
        <div class="result-box">
            <b>Label:</b> {res['label']}<br>
            <b>Confidence:</b> {res['score']:.4f}
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

    bars1 = ax.bar([i - 0.2 for i in x], conf1, width=0.4, label="BERT-Tiny", color="#4CAF50")
    bars2 = ax.bar([i + 0.2 for i in x], conf2, width=0.4, label="RoBERTa", color="#FFC107")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Confidence Score")
    ax.set_title("Perbandingan Confidence Dua Model")
    ax.legend()

    # Tambahkan label angka di atas bar
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 5), textcoords="offset points", ha='center', fontsize=9)

    st.pyplot(fig)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("🧠 Model dari 🤗 Hugging Face | Dibuat oleh Rio Anas | UI by Streamlit")
