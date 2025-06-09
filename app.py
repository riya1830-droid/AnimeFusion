import streamlit as st
import pandas as pd
import pickle

# Load anime dataset
anime = pd.read_csv("top_anime_dataset.csv")  # Make sure this CSV has 'name', 'genres', 'image_url' columns

import numpy as np
data = np.load('similarity_compressed.npz')
similarity = data['similarity']
# Load similarity matrix
#similarity = pickle.load(open('similarity.pkl', 'rb'))
# Recommend function with NaN handling
def recommend(selected_genre):
    # Remove rows where genres are NaN
    clean_anime = anime.dropna(subset=['genres'])

    # Get matching indexes
    matching_indexes = clean_anime[clean_anime['genres'].str.contains(selected_genre, case=False, na=False)].index

    if matching_indexes.empty:
        return [], []

    anime_index = matching_indexes[0]
    distances = similarity[anime_index]
    anime_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_anime = []
    recommended_posters = []

    for i in anime_list:
        name = anime.iloc[i[0]]['name']
        poster = anime.iloc[i[0]]['image_url']
        recommended_anime.append(name)
        recommended_posters.append(poster)

    return recommended_anime, recommended_posters


# UI layout
st.set_page_config(page_title="AnimeFusion", layout="wide")
st.title("⛩️AnimeFusion⛩️")
st.header("Anime Recommendation System")
st.subheader("*Find your next favorite anime 🉐 !*")


#  Add a banner image if you have one
st.image("animepic.jpg", use_container_width=True)

# Genre selection
genres = anime['genres'].dropna().unique()
selected_genre = st.selectbox("Select genre", genres)

if st.button("Recommend✨"):
    names, posters = recommend(selected_genre)

    if names:
        cols = st.columns(5)
        for i in range(len(names)):
            with cols[i]:
                st.write(names[i])
                st.image(posters[i])
                # Print the URL to check it's generated correctly
                mal_url = f"https://myanimelist.net/search/all?q={names[i].replace(' ', '%20')}&cat=all"
                st.write("[Know More About Anime....]", mal_url)
                # Test with a direct link using markdown
                st.markdown(f"[🔗 Open on MAL]({mal_url})", unsafe_allow_html=True)
    else:
        st.warning("No recommendations found for this genre.")






