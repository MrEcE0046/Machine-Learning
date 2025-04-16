import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

def load_data():
    users = pd.read_csv('users.csv')
    books = pd.read_csv('books.csv')
    ratings = pd.read_csv('ratings.csv')
    new_books = books.drop_duplicates('Book-Title')
    ratings_with_name = ratings.merge(new_books, on='ISBN')
    ratings_with_name.drop(["ISBN", "Image-URL-S", "Image-URL-M"], axis = 1, inplace = True)
    ratings_with_name.dropna(inplace = True)

    users_ratings_matrix = ratings_with_name.merge(users, on="User-ID")
    users_ratings_matrix.drop(["Location", "Age"], axis = 1, inplace = True)

    # print(users_ratings_matrix)

    return users_ratings_matrix, new_books

def extract_features(users_ratings_matrix):
    x = users_ratings_matrix.groupby("User-ID").count()["Book-Rating"] > 100
    knowledgeable_users = x[x].index
    filtered_users_ratings = users_ratings_matrix[users_ratings_matrix["User-ID"].isin(knowledgeable_users)]

    y = filtered_users_ratings.groupby("Book-Title").count()["Book-Rating"] > 50
    famous_books = y[y].index
    final_users_ratings = filtered_users_ratings[filtered_users_ratings["Book-Title"].isin(famous_books)]

    pivot_table = final_users_ratings.pivot_table(index = "Book-Title", columns = "User-ID", values = "Book-Rating")
    pivot_table.fillna(0, inplace = True)

    title = pivot_table.index
    # print(final_users_ratings.columns)

    return pivot_table, title

def make_model(pivot_table):
    scaler = StandardScaler(with_mean=True, with_std=True)
    pivot_table_normalized = scaler.fit_transform(pivot_table)
    similarity_score = cosine_similarity(pivot_table_normalized)

    return similarity_score

def recommend(book_name, new_books, similarity_score, pivot_table):
    index = np.where(pivot_table.index == book_name)[0][0]
    similar_books = sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[1:6]
    data = []

    for index,similarity in similar_books:
        items = []
        temp_df = new_books[new_books["Book-Title"]==pivot_table.index[index]]
        items.extend(temp_df["Book-Title"].values)
        items.extend(temp_df["Book-Author"].values)
        items.extend(temp_df["Image-URL-M"].values)
        data.append(items)

    return data

# ==================================================================================================================

def get_poster(similar_books): # pivot_table??
    book_name = []
    IDs_index = []
    poster_url = []

    for book_id in similar_books:
        book_name.append(pivot_table.index[book_id])

    for name in book_name[0]:
        IDs = np.where(new_books["Book-Title"] == name)[0][0]
        IDs_index.append(IDs)

    for idx in IDs_index:
        url = new_books.iloc[idx]["Image-URL-M"]
        poster_url.append(url)

    return poster_url

def app_recommend(book_name, similarity_score, pivot_table):
    book_list = []
    index = np.where(pivot_table.index == book_name)[0][0]
    similar_books = sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[1:6]

    poster_url = get_poster(similar_books) # suggestion

    for i in range(len(similar_books)):
        books = pivot_table.index[similar_books[i]]
        for j in books:
            book_list.append(j)
    
    return book_list, poster_url


def main():
    table, new_books = load_data()
    pivot_table, title = extract_features(table)
    similarity_score = make_model(pivot_table)
    # app(pivot_table)

    # print(table.columns) # innehåller hela datasetet utan bilder - 
    # print(new_books) # Innehåller ISBN, Titel och bilder -
    # print(pivot_table) # innehåller distanser med titlar i första kolumn - book_pivot
    # print(similarity_score) # innehåller std distanser enbart - 
    # print(title) bara boknamn - books_name

    # book_name = "1984"
    # recomendations = recommend(book_name, new_books, similarity_score, pivot_table)

    # print(f"Top recommendations for {book_name}:")
    # for i in recomendations:
    #     print(i


    # st.header("Book Recommendations using ML ")

    # pivot_table = st.selectbox(
    #     "Type or select a book", title
    # )

    # if st.button("Show Recommendation"):
    #     recommendation_books, poster_url = app_recommend(selected_books)
    #     col1, col2, col3, col4, col5, = st.columns(5)

    #     with col1:
    #         st.text(recomendation_books[1])
    #         st.image(poster_url[1])

    #     with col2:
    #         st.text(recomendation_books[2])
    #         st.image(poster_url[2])

    #     with col3:
    #         st.text(recomendation_books[3])
    #         st.image(poster_url[3])

    #     with col4:
    #         st.text(recomendation_books[4])
    #         st.image(poster_url[4])

    #     with col5:
    #         st.text(recomendation_books[5])
    #         st.image(poster_url[5])


if __name__ == "__main__":
    main()