import streamlit as st
from transformers import pipeline

# -------------------------------
# Setup halaman & gaya UI
# -------------------------------
st.set_page_config(page_title="Deteksi Hoaks Indonesia", layout="centered")

st.markdown("""
<style>
.result-box {
    background-color: rgba(255,255,255,0.05);
    padding: 1em;
    border-radius: 10px;
    border: 1px solid #666;
    margin-bottom: 1em;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    color: white;
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
st.write("Bandingkan hasil klasifikasi teks dari dua model NLP: **XLM-RoBERTa** dan **IndoBERT**.")

text_input = st.text_area("📄 Masukkan teks berita atau opini:",
    "Vaksin COVID-19 mengandung chip untuk melacak masyarakat.")

if not text_input.strip():
    st.warning("⚠️ Harap masukkan teks terlebih dahulu.")
    st.stop()

# -------------------------------
# Load Dua Model
# -------------------------------
@st.cache_resource
def load_models():
    pipe_roberta = pipeline("text-classification", model="khavitidala/xlmroberta-large-fine-tuned-indo-hoax-classification")
    pipe_indo = pipeline("text-classification", model="Rifky/indobert-hoax-classification")
    return pipe_roberta, pipe_indo

pipe1, pipe2 = load_models()

# -------------------------------
# Prediksi
# -------------------------------
if st.button("🔍 Deteksi Hoaks"):
    res1 = pipe1(text_input)[0]
    res2 = pipe2(text_input)[0]

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

    for title, label, score in zip(["🔹 XLM-RoBERTa", "🔹 IndoBERT"],
                                    [label1, label2], [score1, score2]):
        st.markdown(f"### {title}")
        st.markdown(f"""
        <div class="result-box">
            <b>Label:</b> {label}<br>
            <b>Confidence:</b> {score:.4f}
        </div>
        """, unsafe_allow_html=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("🧠 Model oleh 🤗 Hugging Face | UI by Streamlit | Dibuat oleh Rio Anas")
