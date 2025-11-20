from fastapi import FastAPI, HTTPException
from typing import List
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()


# -----------------------------
#  DATA LOADING & PREPROCESSING
# -----------------------------

# Load CSVs from /public folder (IMPORTANT for Vercel)
movies = pd.read_csv("../public/tmdb_5000_movies.csv")
credits = pd.read_csv("../public/tmdb_5000_credits.csv")

# Merge on title
movies = movies.merge(credits, on="title")

# Keep only important columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']].copy()

# Drop missing
movies.dropna(inplace=True)


def convert(obj):
    """Convert genres/keywords JSON string -> list of names."""
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


def convert3(obj):
    """Convert cast JSON -> top 3 cast names."""
    L = []
    cnt = 0
    if isinstance(obj, str):
        obj = ast.literal_eval(obj)
    for i in obj:
        if isinstance(i, dict) and 'name' in i:
            L.append(i['name'])
            cnt += 1
        elif isinstance(i, str):
            L.append(i)
            cnt += 1
        if cnt == 3:
            break
    return L


def convert33(obj):
    """Convert crew JSON -> director."""
    L = []
    if isinstance(obj, str):
        obj = ast.literal_eval(obj)
    for i in obj:
        if i.get('job') == 'Director':
            L.append(i['name'])
            break
    return L


def splitt(s: str):
    """Split overview."""
    return s.split()


# Apply transformations
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(convert33)
movies['overview'] = movies['overview'].apply(splitt)

# Remove spaces inside names
for col in ['genres', 'cast', 'crew', 'keywords']:
    movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])

# Create tags
movies['tags'] = (
    movies['overview'] +
    movies['genres'] +
    movies['keywords'] +
    movies['cast'] +
    movies['crew']
)

# New data
newdf = movies[['movie_id', 'title', 'tags']].copy()
newdf['tags'] = newdf['tags'].apply(lambda x: " ".join(x).lower())

# Text vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(newdf['tags']).toarray()

# Cosine similarity
similarity = cosine_similarity(vectors)

# Cache titles
titles = newdf['title'].tolist()


def get_recommendations(movie: str, top_n: int = 5) -> List[str]:
    """Return top_n similar movies."""
    if movie not in newdf['title'].values:
        raise ValueError("Movie not found in database")

    idx = newdf[newdf['title'] == movie].index[0]
    distances = similarity[idx]

    movie_list = sorted(
        enumerate(distances),
        key=lambda x: x[1],
        reverse=True
    )[1:top_n + 1]

    return [newdf.iloc[i[0]].title for i in movie_list]


# -----------------------------
#  API ENDPOINTS
# -----------------------------

@app.get("/")
def root():
    return {"message": "Movie Recommender API is running!"}


@app.get("/recommend")
def recommend(movie: str):
    """Return 5 movie recommendations."""
    try:
        recs = get_recommendations(movie, top_n=5)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {"movie": movie, "recommendations": recs}


@app.get("/movies")
def list_movies():
    """Return first 100 movies."""
    return {"movies": titles[:100]}
