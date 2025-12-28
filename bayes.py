import streamlit as st
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# =========================
# JUDUL APLIKASI
# =========================
st.set_page_config(page_title="Bayesian Spam Classifier", layout="centered")
st.title("📨 Bayesian Spam Classifier")
st.write("Aplikasi sederhana Machine Learning menggunakan **Teorema Bayes (Naive Bayes)**")

# =========================
# DATASET LATIH
# =========================
data_latih = [
    ("gratis hadiah uang", "Spam"),
    ("menang undian gratis", "Spam"),
    ("uang gratis sekarang", "Spam"),
    ("rapat kerja hari ini", "Tidak Spam"),
    ("jadwal meeting kantor", "Tidak Spam"),
    ("laporan kerja selesai", "Tidak Spam"),
]

texts = [x[0] for x in data_latih]
labels = [x[1] for x in data_latih]

# =========================
# TRAIN MODEL
# =========================
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X_train, labels)

# =========================
# INPUT USER
# =========================
st.subheader("✍️ Masukkan Teks Email")
input_text = st.text_area("Tulis pesan di sini:")

# =========================
# PREDIKSI
# =========================
if st.button("🔍 Prediksi"):
    if input_text.strip() == "":
        st.warning("Masukkan teks terlebih dahulu!")
    else:
        X_test = vectorizer.transform([input_text])
        prediction = model.predict(X_test)[0]
        proba = model.predict_proba(X_test)

        st.subheader("📊 Hasil Prediksi")
        st.success(f"Pesan ini diklasifikasikan sebagai: **{prediction}**")

        st.write("Probabilitas:")
        st.write(f"- Spam: {proba[0][list(model.classes_).index('Spam')]:.2f}")
        st.write(f"- Tidak Spam: {proba[0][list(model.classes_).index('Tidak Spam')]:.2f}")

# =========================
# PENJELASAN
# =========================
with st.expander("📘 Penjelasan Teorema Bayes"):
    st.write("""
    Teorema Bayes menghitung probabilitas suatu kelas berdasarkan data masukan:

    P(C|X) = P(X|C) × P(C)

    Dalam aplikasi ini:
    - C = kelas (Spam / Tidak Spam)
    - X = kata-kata dalam teks
    - Model memilih kelas dengan probabilitas terbesar
    """)
