import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Fungsi untuk membersihkan teks
def clean_text(text):
    # Lakukan pembersihan teks sesuai kebutuhan (misalnya, menghilangkan karakter khusus)
    cleaned_text = re.sub(r"[^a-zA-Z]", " ", text)
    return cleaned_text

# Fungsi untuk mengganti kata-kata slang
def replace_slang_words(text):
    # Lakukan penggantian kata-kata slang sesuai kebutuhan
    # Misalnya, Anda dapat memiliki kamus kata-kata slang yang ingin diganti
    return text

# Fungsi untuk memproses data
def process_data(data):
    # Membersihkan teks
    clean_text_data = data.apply(clean_text)

    # Mengganti kata-kata slang
    slang_words = []
    for i in range(len(clean_text_data)):
        slang = replace_slang_words(clean_text_data[i])
        slang_words.append(slang)

    # Menghasilkan token kata-kata tanpa stopwords
    words = []
    for i in range(len(slang_words)):
        tokens = word_tokenize(slang_words[i])
        listStopword = set(stopwords.words('indonesian'))
        removed = [t for t in tokens if t not in listStopword]
        words.append(removed)

    # Menggabungkan kata-kata
    gabung = [' '.join(w) for w in words]

    return gabung

# Memuat data
data = pd.read_csv("https://raw.githubusercontent.com/20-184Bagas/ppw/main/databeritadetikstreamlit.csv")

# Memproses data
gabung = process_data(data['Kalimat Berita'])

# TF-IDF Vectorizer
tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
tfidf_wm = tfidfvectorizer.fit_transform(gabung)

# Membuat DataFrame untuk matriks TF-IDF
tfidf_tokens = tfidfvectorizer.get_feature_names_out()
df_tfidfvect = pd.DataFrame(data=tfidf_wm.toarray(), columns=tfidf_tokens)

# Menampilkan data berita dan matriks TF-IDF
st.title("Aplikasi Klasifikasi Berita")
st.header("Data Berita")
st.dataframe(data[['Kalimat Berita', 'Jenis Berita']])
st.header("Matriks TF-IDF")
st.dataframe(df_tfidfvect)

# Klasifikasi menggunakan Random Forest
X_train = tfidf_wm
y_train = data['Jenis Berita']
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Input pengguna untuk berita yang ingin diklasifikasikan
user_input = st.text_input("Masukkan Berita untuk Klasifikasi:", "")
user_input_cleaned = clean_text(user_input)
user_input_slang = replace_slang_words(user_input_cleaned)
user_input_words = [t for t in word_tokenize(user_input_slang) if t not in set(stopwords.words('indonesian'))]
user_input_gabung = ' '.join(user_input_words)

# Mengubah berita input menjadi matriks TF-IDF
user_input_tfidf = tfidfvectorizer.transform([user_input_gabung])

# Prediksi kategori berita menggunakan model Random Forest
prediction = rf_model.predict(user_input_tfidf)
st.header("Hasil Klasifikasi")
st.write(f"Berita ini tergolong dalam kategori jenis berita: **{prediction[0]}**")
