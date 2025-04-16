from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pandas as pd 
import numpy as np
import re
from scipy.sparse import hstack, csr_matrix
import streamlit as st



def load_data():
    tags = pd.read_csv("tags.csv")
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    links = pd.read_csv("links.csv")
    return tags, movies, ratings, links

def data_prep(tags, movies, links):
    """ Förbereder data inför td-idf vektorisering med enkel data rensning. 
    inspo: https://www.youtube.com/watch?v=eyEabQRBMQA&t=553s&ab_channel=Dataquest """

    movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)
    tags = tags[tags['tag'].apply(lambda x: isinstance(x, str))]
    merged_tags = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(set(x))).reset_index()
    filtered_movies = pd.merge(movies, merged_tags, on="movieId", how="inner")
    filtered_movies = pd.merge(filtered_movies, links, on="movieId", how="inner")

    text_columns = filtered_movies[["genres", "tag"]]

    for col in text_columns:
        filtered_movies[col] = filtered_movies[col].apply(clean_text)

    return filtered_movies

def clean_text(text):
    if type(text)==str:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    return text

def tfidf(filtered_movies):
    """ Vekoriserar både genre och tags kolumnerna och kombinerar dem bredvid varandra inför KNN fit och får ut olika vektorer per film. """

    text_data1 = filtered_movies["genres"].tolist()
    vectorizer1 = TfidfVectorizer()
    tfidf_matrix1 = vectorizer1.fit_transform(text_data1)

    text_data2 = filtered_movies["tag"].tolist()
    vectorizer2 = TfidfVectorizer()
    tfidf_matrix2 = vectorizer2.fit_transform(text_data2) # ändarade vectorizer1 till 2

    combined_matrix = hstack((tfidf_matrix1, tfidf_matrix2))
    return combined_matrix

def model_KNN(combined_matrix):
    model_knn = NearestNeighbors(metric = "cosine", algorithm = "auto")
    model_knn.fit(combined_matrix)
    return model_knn

def rating_features(ratings, movies):
    """ Skalar ned ratings för inte göra beräkningarna för tung, borde man spara ned resultaten i en csv för snabbare körningar? """

    x = ratings["userId"].value_counts() > 200
    y = x[x].index
    ratings = ratings[ratings["userId"].isin(y)]
    ratings_with_movies = ratings.merge(movies, on= "movieId")
    return ratings_with_movies

def titles_to_streamlit(ratings_with_movies):
    return ratings_with_movies["title"].tolist()

def final_rating(ratings_with_movies):
    num_rating = ratings_with_movies.groupby("title")["rating"].count().reset_index()

    num_rating.rename(columns={"rating": "num_of_rating"}, inplace=True)
    final_rating = ratings_with_movies.merge(num_rating, on="title") # sätter ihop dataset som både har title kolumner på title
    final_rating = final_rating.drop(columns=["title", "genres"])
    return final_rating

def get_recommendations(movie_name, combined_matrix, model_knn, filtered_movies, top_n = 50): # simularity, spara en featuremängd, get_features

    movie_id = filtered_movies[filtered_movies["title"] == movie_name].index[0]
    movie = combined_matrix[movie_id]
    
    distances, suggestions = model_knn.kneighbors(movie.reshape(1, -1), n_neighbors= top_n+1)
    
    return filtered_movies[["movieId","title", "genres", "tmdbId"]].iloc[suggestions[0][0:top_n+1]]

def top50_with_ratings(list, final_rating):
    top50_n_ratings = pd.merge(list, final_rating, on="movieId", how="inner")
    return top50_n_ratings

def pivot(top50_n_ratings):
    """ Funderade på att nöja mig här genom att ta ut 5 filmer med flest ratings och högst medel på rating. Men var nyfiken på hur en obevakad nearest neighbor 
    presterar med givet 50 filmer och dess rating. 
    Funktionen parar ihop givet 50 titlar med ratings och skapar en pivottabell och lägger till nollor på titlar från användare som inte har bedömt dem.
    Då pivottabellen innehåller många nollar skapas en komprimerad gles matris för minnesoptimering. """

    movie_pivot = top50_n_ratings.pivot_table(columns="userId", index="title", values="rating")
    movie_pivot.fillna(0, inplace=True)
    return movie_pivot

def sparse(movie_pivot):
    movie_sparse = csr_matrix(movie_pivot)
    return movie_sparse

def movie_names_to_streamlit(movie_pivot):
    movie_names = movie_pivot.index.tolist() # Gjorde en lista för streamlit
    return movie_names

def recommend(movie_name, movie_pivot, model):
    movie_list = []

    movie_id = np.where(movie_pivot.index == movie_name)[0][0]
    distance, suggestion = model.kneighbors(movie_pivot.iloc[movie_id,:].values.reshape(1,-1), n_neighbors=6) 
    
    # poster_url = get_poster(suggestion, book_pivot, final_rating)
    
    for i in range(len(suggestion)):
        movies = movie_pivot.index[suggestion[i]]
        for j in movies:
            movie_list.append(j)
    return movie_list

def get_rec(combined_matrix, model_knn, filtered_movies, final_rating, titles):

    st.title("Movie Recommender by LM")

    user_input = st.selectbox("Type or select a movie", titles)
    
    if st.button("Show Recommendeation"):
        list = get_recommendations(user_input, combined_matrix, model_knn, filtered_movies, top_n = 50)
        top50_n_ratings = top50_with_ratings(list, final_rating)
        movie_pivot = pivot(top50_n_ratings)
        movie_sparse = sparse(movie_pivot)
        model = model_KNN(movie_sparse)
        x = recommend(user_input, movie_pivot, model)
        col1, col2, col3, col4, col5, = st.columns(5)

        with col1:
            st.text(x[1])
        with col2:
            st.text(x[2])
        with col3:
            st.text(x[3])
        with col4:
            st.text(x[4])
        with col5:
            st.text(x[5])

def main():
    tags, movies, ratings, links = load_data()
    filtered_movies = data_prep(tags, movies, links)
    combined_matrix = tfidf(filtered_movies)
    model_knn = model_KNN(combined_matrix)
    ratings_with_movies = rating_features(ratings, movies)
    titles = titles_to_streamlit(ratings_with_movies)
    final_ratings = final_rating(ratings_with_movies)
    get_rec(combined_matrix, model_knn, filtered_movies, final_ratings, titles)


if __name__ == "__main__":
   main()