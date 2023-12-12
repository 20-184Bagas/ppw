import requests
from bs4 import BeautifulSoup
import streamlit as st
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download punkt tokenizer untuk nltk
nltk.download('punkt')

# Fungsi untuk mendapatkan teks dari URL
def get_article_text(article_url):
    response = requests.get(article_url)
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    article = soup.find('div', class_="post-content clearfix")
    return article.get_text() if article else ''

# Mendapatkan teks dari URL
url = "https://www.antaranews.com/berita/3582189/tekuk-inter-pada-final-liga-champions-city-sukses-raih-treble-winners?utm_source=antaranews&utm_medium=desktop&utm_campaign=related_news"
article_text = get_article_text(url)

# Tokenisasi teks menjadi kalimat menggunakan nltk
sentences = nltk.sent_tokenize(article_text)

# Menampilkan kalimat-kalimat
st.header("Kalimat-kalimat dari Artikel Berita")
for sentence in sentences:
    st.write(sentence)

# Hitung TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_values = tfidf_matrix.toarray()

# Menampilkan hasil TF-IDF untuk setiap kata dalam setiap kalimat
st.header("TF-IDF untuk Setiap Kata dalam Setiap Kalimat")
for i, sentence in enumerate(sentences):
    st.subheader(f"Kalimat {i + 1}: {sentence}")
    for j, word in enumerate(feature_names):
        tfidf_value = tfidf_values[i][j]
        if tfidf_value > 0:
            st.write(f"{word}: {tfidf_value:.4f}")

# Hitung cosine similarity antara semua pasangan kalimat
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Buat DataFrame dari hasil cosine similarity
df_similarity = pd.DataFrame(similarity_matrix, columns=sentences, index=sentences)

# Menampilkan matriks similarity
st.header("Matriks Similarity antara Kalimat-kalimat")
st.write(df_similarity)

# Buat grafik matriks similarity
fig, ax = plt.subplots()
cax = ax.matshow(df_similarity, cmap='coolwarm')
fig.colorbar(cax)

# Memberi label pada sumbu X dan Y
ax.set_xticks(np.arange(len(df_similarity.columns)))
ax.set_yticks(np.arange(len(df_similarity.index)))
ax.set_xticklabels(df_similarity.columns, rotation=90)
ax.set_yticklabels(df_similarity.index)

# Menampilkan nilai similarity pada matriks
for i in range(len(df_similarity.index)):
    for j in range(len(df_similarity.columns)):
        text = ax.text(j, i, f'{df_similarity.iloc[i, j]:.2f}', ha='center', va='center', color='w')

st.pyplot(fig)

# Buat grafik berarah (DiGraph) berdasarkan similarity_matrix
G = nx.DiGraph()
for i in range(len(similarity_matrix)):
    G.add_node(i)  # Tambahkan node dengan indeks numerik

for i in range(len(similarity_matrix)):
    for j in range(len(similarity_matrix)):
        similarity = similarity_matrix[i][j]
        if similarity > 0 and i != j:  # Pastikan node tidak menghubungkan dirinya sendiri
            G.add_edge(i, j)

# Hitung closeness centrality
closeness_centrality = nx.closeness_centrality(G, distance='weight')

# Tampilkan closeness centrality
st.header("Closeness Centrality untuk Setiap Kalimat")
for i, (sentence, centrality) in enumerate(closeness_centrality.items()):
    st.write(f"Kalimat {i + 1}: {sentence} - Closeness Centrality: {centrality:.4f}")

# Urutkan node berdasarkan closeness centrality
sorted_nodes = sorted(closeness_centrality, key=closeness_centrality.get, reverse=True)

# Ambil 3 node teratas
top_nodes = sorted_nodes[:3]

data = []
for node in top_nodes:
    sentence = sentences[node]
    centrality = closeness_centrality[node]
    data.append((f"{node}", centrality, sentence))

df_closeness_centrality = pd.DataFrame(data, columns=["Rank (Node)", "Closeness Centrality", "Sentence"])

# Menampilkan DataFrame closeness centrality
st.header("DataFrame Closeness Centrality")
st.write(df_closeness_centrality)
