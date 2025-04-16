import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_data():
    books = pd.read_csv("Books.csv")
    ratings = pd.read_csv("Ratings.csv")
    books.drop_duplicates("Book-Title", inplace=True)
    merge = ratings.merge(books, on= "ISBN")
    merge.drop(["ISBN", "Image-URL-S", "Image-URL-M", "Image-URL-L"], axis=1, inplace=True)
    merge.dropna(inplace=True)
    return merge

""" #%% """
def extract_features(raw_table):
    x = raw_table.groupby("User-ID").count()["Book-Rating"] > 100
    expert_users = x[x].index

    filtered_ratings = raw_table[raw_table["User-ID"].isin(expert_users)]
    y = filtered_ratings.groupby("Book-Title").count()["Book-Rating"] > 50
    famous_books = y[y].index

    user_ratings = filtered_ratings[filtered_ratings["Book-Title"].isin(famous_books)]

    design_matrix = user_ratings.pivot_table(index="Book-Title", columns="User-ID", values="Book-Rating")
    design_matrix.fillna(0, inplace=True)

    return design_ratings

def make_model():
    table = load_data_files()
    matrix = extract_features(table)
    scaler = standard_scaler(with_mean= True, with_std= True)
    scaled = scaler.fit_transform(matrix)
    sim_score = cosine_similarity(scaled)
    return sim_score

def recommend(book_name, design_matrix, similarity_score):
    index = np.where(matrix.index == book_name)[0][0]
    similar_books = sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)
    data = []

    for i in similar_books[1:6]:
        item = []
        temp_df = table[books["Book-Title"] == design.matrix.index[index]]
        item.extend(temp_df["Book-Title"].values)
        item.extend(temp_df["Book-Author"].values)
        data.append(item)
    return data

def main():
    table = load_data_files()
    matrix = extract_features(table)
    model = make_model(matrix)
    name = "1984"
    print(recommend(name, table, matrix, model))

if __name__ == "__main__":
    main()