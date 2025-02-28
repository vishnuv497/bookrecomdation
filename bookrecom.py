import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv('books.csv')
    df = df[['Book-Title', 'Book-Description']].dropna()
    return df

# Recommend Books
def recommend_books(book_title, data, similarity_matrix):
    if book_title not in data['Book-Title'].values:
        st.error("Book not found. Please check the title.")
        return []

    index = data[data['Book-Title'] == book_title].index[0]
    similarity_scores = list(enumerate(similarity_matrix[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similar_books = [data['Book-Title'].iloc[i[0]] for i in similarity_scores[1:6]]
    return similar_books

# Main App
st.title("Book Recommendation System")
st.write("Get book recommendations based on your favorite book!")

# Load Data
df = load_data()

# Vectorize Descriptions
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Book-Description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# User Input
book_list = df['Book-Title'].values
selected_book = st.selectbox("Select or type a book title:", book_list)

if st.button("Recommend"):
    recommendations = recommend_books(selected_book, df, cosine_sim)
    if recommendations:
        st.subheader("You might also like:")
        for i, book in enumerate(recommendations):
            st.write(f"{i+1}. {book}")
