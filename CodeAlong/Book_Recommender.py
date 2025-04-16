import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import streamlit as st
from scipy.sparse import csr_matrix

""" Kräver installation av streamlit 
kör appen via din terminal med kommandot: run Book_Recommender.py """

def load_data():
    users = pd.read_csv("Users.csv")
    books = pd.read_csv("Books.csv", dtype={"Year-Of-Publication": str})
    ratings = pd.read_csv('Ratings.csv')


    books = books[["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher", "Image-URL-L"]]
    books.rename(columns={
        "Book-Title": "title",
        "Book-Author": "author",
        "Year-Of-Publication": "year",
        "Publisher": "publisher",
        "Image-URL-L": "img_url"},inplace= True)

    ratings.rename(columns={
        "User-ID": "user-id",
        "Book-Rating": "ratings"}, inplace= True)
    
    return books, ratings

def extract_features(books, ratings):
    
    x = ratings["user-id"].value_counts() > 200
    y = x[x].index
    ratings = ratings[ratings["user-id"].isin(y)]
    ratings_with_books = ratings.merge(books, on= "ISBN")

    num_rating = ratings_with_books.groupby("title")["ratings"].count().reset_index()

    num_rating.rename(columns={"ratings": "num_of_rating"}, inplace=True)
    final_rating = ratings_with_books.merge(num_rating, on="title") # sätter ihop dataset som både har title kolumner på title
    final_rating = final_rating[final_rating["num_of_rating"]>50] # tar bort alla böcker med mindre än 50 ratings
    final_rating.drop_duplicates(["user-id", "title"], inplace=True) # tar bort alla dubletter

    book_pivot = final_rating.pivot_table(columns="user-id", index="title", values="ratings")
    book_pivot.fillna(0, inplace=True)
    
    book_sparse = csr_matrix(book_pivot) # Används för att hantera glesa matriser, där många element är nollor. Csr sparar bara de som inte är nollor och gör att matrisen blir mycket smalare och snabbare. 

    return book_sparse, book_pivot, final_rating

def make_model(book_sparse, book_pivot):
    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(book_sparse)

    book_names = book_pivot.index.tolist() # Gjorde till en lista för streamlit

    return book_names, model

def recommend(book_names, book_pivot, model, final_rating):
    book_list = []

    book_id = np.where(book_pivot.index == book_names)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6) 
    
    poster_url = get_poster(suggestion, book_pivot, final_rating)
    
    for i in range(len(suggestion)):
        books = book_pivot.index[suggestion[i]]
        for j in books:
            book_list.append(j)
    
    return book_list, poster_url

def get_poster(suggestion, book_pivot, final_rating):
    """ Fick en bättre funktion av GPT """
    poster_url = []

    for book_id in suggestion[0]:  # Loopa över första arrayen i suggestion
        book_name = book_pivot.index[book_id]
        book_row = final_rating[final_rating["title"] == book_name]
        if not book_row.empty:
            poster_url.append(book_row.iloc[0]["img_url"])

    return poster_url

def streamlit(book_names, book_pivot, model, final_rating):
    st.title("Book Recommender by ML")

    selected_books = st.selectbox("Type or select a book", book_names)

    if st.button("Show Recommendation"):
        book_list, poster_url = recommend(selected_books, book_pivot, model, final_rating)
        col1, col2, col3, col4, col5, = st.columns(5)

        with col1:
            st.text(book_list[1])
            st.image(poster_url[1])
        with col2:
            st.text(book_list[2])
            st.image(poster_url[2])
        with col3:
            st.text(book_list[3])
            st.image(poster_url[3])
        with col4:
            st.text(book_list[4])
            st.image(poster_url[4])
        with col5:
            st.text(book_list[5])
            st.image(poster_url[5])

def main():
    books, ratings = load_data()
    book_sparse, book_pivot, final_rating = extract_features(books, ratings)
    book_names, model = make_model(book_sparse, book_pivot)
    streamlit(book_names, book_pivot, model, final_rating)

if __name__ == "__main__":
    main()