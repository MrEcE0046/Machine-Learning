import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import unicodedata
import re
from functools import cache
from requests import get
import streamlit as st



def load_data():
    tags = pd.read_csv("tags.csv")
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")

    # movies["genres"] = movies["genres"].str.replace("|", " ", regex=False) <--
    # filtererad_data = tags[tags["tag"].str.contains("^[a-zA-Z]+$", na=False)]
    # tags = tags[tags['tag'].apply(lambda x: isinstance(x, str))]
    merged_tags = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(set(x))).reset_index()
    filtered_movies = pd.merge(movies, merged_tags, on="movieId", how="inner")
    filtered_movies["tfidf"] = filtered_movies["tfidf"].apply(lambda x: x.lower())


    return filtered_movies, ratings

def clean_text(text):
    """ Fick funktionen från GPT """
    # Normalisera till NFKD-form och filtrera bort icke-ASCII-tecken
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()
    
    # Ta bort icke-bokstäver, siffror och mellanslag
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Ta bort extra mellanslag
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

@st.cache_data
def extract_features(movies, ratings):
    x = ratings["userId"].value_counts() > 200
    y = x[x].index
    ratings = ratings[ratings["userId"].isin(y)]
    ratings_with_movies = ratings.merge(movies, on="movieId")

    num_rating = ratings_with_movies.groupby("title")["rating"].count().reset_index()
    num_rating.rename(columns={"rating": "num_of_rating"}, inplace=True)

    final_rating = ratings_with_movies.merge(num_rating, on="title")
    final_rating = final_rating[final_rating["num_of_rating"] > 50]
    final_rating.drop_duplicates(["userId", "title"], inplace=True)

    movie_pivot = final_rating.pivot_table(columns="userId", index="title", values="rating")
    movie_pivot.fillna(0, inplace=True)

    movie_sparse = csr_matrix(movie_pivot)

    return movie_sparse, movie_pivot, final_rating

def make_model(movie_sparse, movie_pivot):
    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(movie_sparse)
    movie_names = movie_pivot.index.tolist()

    return movie_names, model

import numpy as np

# def recommend(movie_names, book_pivot, model, final_rating): med posters?

def recommend(selected_movies, movie_pivot, model):
    movie_list = []

    movie_id = np.where(movie_pivot.index == selected_movies)[0][0]
    distance, suggestion = model.kneighbors(movie_pivot.iloc[movie_id,:].values.reshape(1,-1), n_neighbors = 6)
    
    for i in range(len(suggestion)):
        movies = movie_pivot.index[suggestion[i]]
        for j in movies:
            movie_list.append(j)

    return movie_list

"""======================================================================================================================"""
def recommend_with_poster(selected_movie, movie_pivot, model, final_rating, movies):

    index = movies[movies["title"] == selected_movie].index[0]
    distance = sorted(list(enumerate(simularity_score[index]))), reverse= True, key= lambda x: x[1]
    movie_list = []
    poster = []
    for i in distance[1:6]:
        movie_id = movies.iloc[i[0]].movieId
        movie_list.append(movies.iloc[i[0]].title)
        poster.append(get_poster(movie_id))

    return movie_list, poster

def get_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=df95d734039c3034368dbc0d3f33068c&language=en-US".format(movie_id)
    data = data.get(url)
    data = data.json()
    get_poster = data["poster_path"]

    return f"https://image.tmdb.org/t/p/w500/{get_poster}" 
"""======================================================================================================================"""

def streamlit(movie_names, movie_pivot, model, poster):
    st.title("Movie Recommender by LM")

    selected_movies = st.selectbox("Type or select a movie", movie_names)

    if st.button("Show Recommendeation"):
        movie_list = recommend(selected_movies, movie_pivot, model)
        col1, col2, col3, col4, col5, = st.columns(5)
        
        with col1:
            st.text(movie_list[0])
            st.image(poster[0])
        with col2:
            st.text(movie_list[1])
            st.image(poster[1])
        with col3:
            st.text(movie_list[2])
            st.image(poster[2])
        with col4:
            st.text(movie_list[3])
            st.image(poster[3])
        with col5:
            st.text(movie_list[4])
            st.image(poster[4])

def main():
    movies, ratings = load_data()
    movie_sparse, movie_pivot, final_rating = extract_features(movies, ratings)
    movie_names, model = make_model(movie_sparse, movie_pivot)
    streamlit(movie_names, movie_pivot, model)

if __name__ == "__main__":
    main()