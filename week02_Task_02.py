# week02_Task_02.py

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit App Title
st.title("🎬 Movie Recommendation System")
st.markdown("Get similar movie recommendations based on Overview, Genre, and Title.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("F:/AWFERA/Machine learning/MachineLearningInternshipProject/mymoviedb.csv",
                     encoding='utf-8',
                     engine='python',
                     on_bad_lines='skip')  # Skips bad rows
    df.fillna('', inplace=True)
    df['combined_features'] = df['Overview'] + ' ' + df['Genre'] + ' ' + df['Title']
    return df

df = load_data()

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

# Cosine Similarity Matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Movie Recommendation Function
def recommend_movie(title, num=5):
    title = title.lower()
    df['lower_title'] = df['Title'].str.lower()

    if title not in df['lower_title'].values:
        return ["❌ Movie not found in database."]

    idx = df[df['lower_title'] == title].index[0]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:num+1]
    movie_indices = [i[0] for i in similarity_scores]
    return df[['Title', 'Poster_Url']].iloc[movie_indices].values.tolist()

# Input from user
movie_name = st.text_input("Enter a movie title:")

if st.button("Recommend"):
    if movie_name:
        recommendations = recommend_movie(movie_name)
        if recommendations[0] == "❌ Movie not found in database.":
            st.error(recommendations[0])
        else:
            st.subheader("📽 Recommended Movies:")
            for title, poster in recommendations:
                st.markdown(f"**{title}**")
                if poster:
                    st.image(poster, width=150)
    else:
        st.warning("Please enter a movie title.")
