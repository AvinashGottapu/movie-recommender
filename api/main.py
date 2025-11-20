from fastapi import FastAPI, HTTPException
from typing import List
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = FastAPI()

# -----------------------------
#  DATA LOADING & PREPROCESSING
# -----------------------------

CSV_PATH = os.path.join("public")

movies = pd.read_csv("../public/tmdb_500_movies_subset.csv")
credits = pd.read_csv("../public/tmdb_500_credits_subset.csv")

movies = movies.merge(credits, on="title")

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']].copy()
movies.dropna(inplace=True)


def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]


def convert3(obj):
    L = []
    obj = ast.literal_eval(obj)
    for i in obj[:3]:
        L.append(i['name'])
    return L


def convert33(obj):
    obj = ast.literal_eval(obj)
    for i in obj:
        if i.get('job') == 'Director':
            return [i['name']]
    return [""]


def splitt(s):
    return s.split()


movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(convert33)
movies['overview'] = movies['overview'].apply(splitt)

for col in ['genres', 'keywords', 'cast', 'crew']:
    movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])

movies['tags'] = (
        movies['overview']
        + movies['genres']
        + movies['keywords']
        + movies['cast']
        + movies['crew']
)

newdf = movies[['movie_id', 'title', 'tags']].copy()
newdf['tags'] = newdf['tags'].apply(lambda x: " ".join(x).lower())

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(newdf['tags']).toarray()

similarity = cosine_similarity(vectors)

titles = newdf['title'].tolist()


def get_recommendations(movie: str, top_n: int = 5) -> List[str]:
    if movie not in newdf['title'].values:
        raise ValueError("Movie not found")

    idx = newdf[newdf['title'] == movie].index[0]
    distances = similarity[idx]

    movie_list = sorted(
        enumerate(distances),
        key=lambda x: x[1],
        reverse=True,
    )[1: top_n + 1]

    return [newdf.iloc[i[0]].title for i in movie_list]


@app.get("/")
def home():
    return {"status": "API Running"}


@app.get("/recommend")
def recommend(movie: str):
    try:
        return {"movie": movie, "recommendations": get_recommendations(movie)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/movies")
def movies_list():
    return {"movies": titles[:100]}
