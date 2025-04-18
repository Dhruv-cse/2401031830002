import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load the data
movies = pd.read_csv("/content/tmdb_5000_movies.csv")
credits = pd.read_csv("/content/tmdb_5000_credits.csv")

# 2. Merge on 'title'
movies = movies.merge(credits, on='title')

# 3. Select important features
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# 4. Helper function to convert stringified lists
def parse_list(data):
    try:
        return [i['name'] for i in ast.literal_eval(data)]
    except:
        return []

# 5. Extract director name
def get_director(data):
    try:
        for i in ast.literal_eval(data):
            if i['job'] == 'Director':
                return [i['name']]
        return []
    except:
        return []

# 6. Apply parsing
movies['genres'] = movies['genres'].apply(parse_list)
movies['keywords'] = movies['keywords'].apply(parse_list)
movies['cast'] = movies['cast'].apply(lambda x: parse_list(x)[:3])
movies['crew'] = movies['crew'].apply(get_director)

# 7. Combine all tags into one string
movies['overview'] = movies['overview'].fillna('')
movies['tags'] = movies['overview'] + ' ' + \
                 movies['genres'].apply(lambda x: ' '.join(x)) + ' ' + \
                 movies['keywords'].apply(lambda x: ' '.join(x)) + ' ' + \
                 movies['cast'].apply(lambda x: ' '.join(x)) + ' ' + \
                 movies['crew'].apply(lambda x: ' '.join(x))

# 8. Vectorization with TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['tags'])

# 9. Compute similarity
similarity = cosine_similarity(tfidf_matrix)

# 10. Recommendation function
def recommend(movie_title):
    if movie_title not in movies['title'].values:
        return ["Movie not found."]
    
    idx = movies[movies['title'] == movie_title].index[0]
    distances = list(enumerate(similarity[idx]))
    sorted_movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    
    return [movies.iloc[i[0]].title for i in sorted_movies]

# 11. Run
movie_input = input("Enter a movie title: ")
recommendations = recommend(movie_input)

print("\nRecommended Movies:")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec}")
