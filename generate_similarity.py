import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load your dataset
anime = pd.read_csv("top_anime_dataset.csv")

# Drop rows with NaN in 'genres'
anime = anime.dropna(subset=['genres'])

# Convert genres into feature vectors
cv = CountVectorizer()
genre_matrix = cv.fit_transform(anime['genres'])

# Compute similarity matrix
similarity = cosine_similarity(genre_matrix)

# Save the similarity matrix
with open("similarity.pkl", "wb") as f:
    pickle.dump(similarity, f)

print("✅ similarity.pkl file has been created successfully.")
