import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import streamlit as st
from scipy.sparse import csr_matrix

def load_data():
    users = pd.read_csv('Users.csv')
    books = pd.read_csv("Books.csv", dtype={"Year-Of-Publication": str})
    ratings = pd.read_csv('Ratings.csv')

    books = books[["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher", "Image-URL-L"]]
    books.rename(columns={
        "Book-Title": "title",
        "Book-Author": "author",
        "Year-Of-Publication": "year",
        "Publisher": "publisher",
        "Image-URL-L": "img_url"}, inplace=True)

    ratings.rename(columns={
        "User-ID": "user-id",
        "Book-Rating": "ratings"}, inplace=True)
    
    return books, ratings

def extract_features(books, ratings):
    x = ratings["user-id"].value_counts() > 200
    y = x[x].index
    ratings = ratings[ratings["user-id"].isin(y)]
    
    ratings_with_books = ratings.merge(books, on="ISBN")
    num_rating = ratings_with_books.groupby("title")["ratings"].count().reset_index()
    num_rating.rename(columns={"ratings": "num_of_rating"}, inplace=True)
    
    final_rating = ratings_with_books.merge(num_rating, on="title")
    final_rating = final_rating[final_rating["num_of_rating"] > 50]
    final_rating.drop_duplicates(["user-id", "title"], inplace=True)
    
    book_pivot = final_rating.pivot_table(columns="user-id", index="title", values="ratings")
    book_pivot.fillna(0, inplace=True)
    
    book_sparse = csr_matrix(book_pivot)

    return book_sparse, book_pivot, final_rating

def make_model(book_sparse, book_pivot):
    model = NearestNeighbors(algorithm="brute")
    model.fit(book_sparse)

    return book_pivot.index.tolist(), model  # Konvertera till lista för Streamlit

def recommend(book_name, book_pivot, model, final_rating):
    book_list = []

    if book_name not in book_pivot.index:
        return [], []

    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

    poster_url = get_poster(suggestion, book_pivot, final_rating)
    
    for i in suggestion[0]:  # Första raden i suggestion är själva indexen
        book_list.append(book_pivot.index[i])
    
    return book_list, poster_url

def get_poster(suggestion, book_pivot, final_rating):
    poster_url = []

    for book_id in suggestion[0]:  # Loopa över första arrayen i suggestion
        book_name = book_pivot.index[book_id]
        book_row = final_rating[final_rating["title"] == book_name]
        if not book_row.empty:
            poster_url.append(book_row.iloc[0]["img_url"])

    return poster_url

def main():
    books, ratings = load_data()
    book_sparse, book_pivot, final_rating = extract_features(books, ratings)
    book_names, model = make_model(book_sparse, book_pivot)

    st.title("Book Recommender using ML")

    selected_book = st.selectbox("Type or select a book", book_names)  # Nu en lista, inte en Pandas-index

    if st.button("Show Recommendation"):
        book_list, poster_url = recommend(selected_book, book_pivot, model, final_rating)

        if book_list:
            cols = st.columns(min(len(book_list), 5))  # Dynamiskt antal kolumner

            for i, col in enumerate(cols):
                with col:
                    st.text(book_list[i])
                    st.image(poster_url[i])
        else:
            st.write("No recommendations found. Try another book!")

if __name__ == "__main__":
    main()
