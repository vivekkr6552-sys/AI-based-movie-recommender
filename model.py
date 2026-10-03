import os
import ast
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_and_preprocess():  
    # Get the directory where model.py is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Dynamically build relative paths to CSV files
    movies_path = os.path.join(base_dir, 'data', 'tmdb', 'tmdb_5000_movies.csv')
    credits_path = os.path.join(base_dir, 'data', 'tmdb', 'tmdb_5000_credits.csv')

    # Load Datasets
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    
    # Merge on title
    movies = movies.merge(credits, on='title')
    
    # Extract relevant columns
    data = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'budget', 'revenue', 'vote_average']].copy()
    data.dropna(subset=['overview'], inplace=True)
    
    def convert(obj):
        try:
            return [i['name'] for i in ast.literal_eval(obj)]
        except Exception:
            return []

    def get_director(obj):
        try:
            for i in ast.literal_eval(obj):
                if i.get('job') == 'Director':
                    return i.get('name')
        except Exception:
            return "Unknown"
        return "Unknown"

    def convert_cast(obj):
        try:
            return [i['name'] for i in ast.literal_eval(obj)[:3]]
        except Exception:
            return []

    data['director'] = data['crew'].apply(get_director)
    data['genres_list'] = data['genres'].apply(convert)
    data['keywords_list'] = data['keywords'].apply(convert)
    data['cast_list'] = data['cast'].apply(convert_cast)
    
    # Processing text tags for vectorizer
    data['overview_list'] = data['overview'].apply(lambda x: str(x).split())
    data['tags'] = data['overview_list'] + data['genres_list'] + data['keywords_list'] + data['cast_list']
    data['tags'] = data['tags'].apply(lambda x: " ".join(x).lower())
    
    return data

def compute_similarity(df):
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    vectors = tfidf.fit_transform(df['tags']).toarray()
    similarity = cosine_similarity(vectors)
    return similarity

def recommend(movie_title, df, similarity):
    if movie_title not in df['title'].values:
        return []
    
    index = df[df['title'] == movie_title].index[0]
    distances = similarity[index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommendations = []
    for i in movies_list:
        movie_data = df.iloc[i[0]]
        recommendations.append({
            'movie_id': movie_data['movie_id'],
            'title': movie_data['title'],
            'cast': ", ".join(movie_data['cast_list']),
            'director': movie_data['director'],
            'budget': movie_data['budget'],
            'revenue': movie_data['revenue'],
            'rating': movie_data['vote_average'],
            'overview': movie_data['overview']
        })
    return recommendations
