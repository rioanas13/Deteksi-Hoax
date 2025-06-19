import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt

# ------------------------------------
# Setup halaman & gaya
# ------------------------------------
st.set_page_config(page_title="Deteksi Hoaks Indonesia", layout="centered")

st.markdown("""
<style>
.result-box {
    background-color: #f1f3f4;
    padding: 1em;
    border-radius: 10px;
    border: 1px solid #ccc;
    margin-bottom: 1em;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}
.stButton>button {
    background-color: #1f8b4c;
    color: white;
    border-radius: 8px;
    padding: 0.5em 1em;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Deteksi Hoaks Berbahasa Indonesia")
st.write("Bandingkan hasil klasifikasi dari dua model NLP: **XLM-RoBERTa** dan **IndoBERT**.")

text_input = st.text_area("📄 Masukkan teks berita atau opini:", 
    "Vaksin COVID-19 mengandung chip untuk melacak masyarakat.")

if not text_input.strip():
    st.warning("⚠️ Masukkan teks terlebih dahulu.")
    st.stop()

# ------------------------------------
# Load Dua Model dengan Caching
# ------------------------------------
@st.cache_resource
def load_models():
    pipe_roberta = pipeline("text-classification", model="khavitidala/xlmroberta-large-fine-tuned-indo-hoax-classification")
    pipe_indo = pipeline("text-classification", model="Rifky/indobert-hoax-classification")
    return pipe_roberta, pipe_indo

pipe1, pipe2 = load_models()

# ------------------------------------
# Prediksi
# ------------------------------------
if st.button("🔍 Deteksi Hoaks"):
    res1 = pipe1(text_input)[0]
    res2 = pipe2(text_input)[0]

    # Mapping label: baik jika string atau integer
    label_map = {
        "0": "HOAX", "1": "NON-HOAX",
        0: "HOAX", 1: "NON-HOAX",
        "LABEL_0": "HOAX", "LABEL_1": "NON-HOAX"
    }

    label1 = label_map.get(res1['label'], str(res1['label']))
    label2 = label_map.get(res2['label'], str(res2['label']))

    score1 = res1['score']
    score2 = res2['score']

    st.subheader("📊 Hasil Prediksi")

    for title, label, score in zip(["🔹 XLM-RoBERTa", "🔹 IndoBERT"], [label1, label2], [score1, score2]):
        st.markdown(f"### {title}")
        st.markdown(f"""
        <div class="result-box">
            <b>Label:</b> {label}<br>
            <b>Confidence:</b> {score:.4f}
        </div>
        """, unsafe_allow_html=True)

    # ------------------------------------
    # Visualisasi Confidence
    # ------------------------------------
    st.markdown("### 📈 Perbandingan Confidence")

    labels = ["HOAX", "NON-HOAX"]
    conf1 = [score1 if label1 == l else 1 - score1 for l in labels]
    conf2 = [score2 if label2 == l else 1 - score2 for l in labels]

    fig, ax = plt.subplots()
    x = range(len(labels))

    bars1 = ax.bar([i - 0.2 for i in x], conf1, width=0.4, label="XLM-RoBERTa", color="#4CAF50")
    bars2 = ax.bar([i + 0.2 for i in x], conf2, width=0.4, label="IndoBERT", color="#FFC107")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Confidence Score")
    ax.set_title("Perbandingan Confidence Dua Model")
    ax.legend()

    # Tambahkan angka confidence di atas bar
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 4), textcoords="offset points", ha='center', fontsize=9)

    st.pyplot(fig)

# ------------------------------------
# Footer
# ------------------------------------
st.markdown("---")
st.caption("📦 Model oleh 🤗 Hugging Face | UI by Streamlit | Dibuat oleh Rio Anas")
