import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# 1. Load Dataset
df = pd.read_csv("tmdb_5000_movies.csv")
df = df[['title', 'overview', 'genres', 'keywords']]

# 2. Preprocess and Combine Text Columns
df['overview'] = df['overview'].fillna('')
df['genres'] = df['genres'].fillna('')
df['keywords'] = df['keywords'].fillna('')
df['content'] = df['overview'] + ' ' + df['genres'] + ' ' + df['keywords']

# 3. TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content'])

# 4. Cosine Similarity Matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# 5. Title to Index Mapping
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# 6. Recommendation Function
def recommend(title, cosine_sim=cosine_sim):
    if title not in indices:
        return ["Movie not found."]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

# 7. Example Usage
user_input = input("Enter a movie title: ")
recommendations = recommend(user_input)
print("\nRecommended Movies:")
for i, movie in enumerate(recommendations, 1):
    print(f"{i}. {movie}")
