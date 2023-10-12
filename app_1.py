import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from joblib import load
from scipy.spatial.distance import cdist
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict

# Load data and trained model
data = pd.read_csv("data.csv")
song_cluster_pipeline = load('song_cluster_model.joblib')

# Spotify authentication
CLIENT_ID = "370b75e62a77439c87fd0b012fb55942"
CLIENT_SECRET = "0f18e2ae425240b186b5c3e3ec38fdaa"
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET))

# Define the numerical columns used for clustering
number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

# Functions to fetch song details from Spotify
def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q='track: {} year: {}'.format(name, year), limit=1)
    if results['tracks']['items'] == []:
        return None
    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]
    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]
    for key, value in audio_features.items():
        song_data[key] = value
    return pd.DataFrame(song_data)

def get_song_data(song, spotify_data):
    try:
        if 'year' in song:
            song_data = spotify_data[(spotify_data['name'] == song['name']) & (spotify_data['year'] == song['year'])].iloc[0]
        else:
            song_data = spotify_data[spotify_data['name'] == song['name']].iloc[0]
        return song_data
    except IndexError:
        return find_song(song['name'], song.get('year'))


# Recommendation functions
def get_mean_vector(song_list, spotify_data):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)
    song_matrix = np.array(song_vectors)
    return np.mean(song_matrix, axis=0)

def recommend_songs_ui(song_list, spotify_data, n_songs=10):
    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    rec_songs = spotify_data.iloc[index]
    return rec_songs

# Streamlit UI
st.title('Music Recommendation System')

# Dropdown for song selection
song_names = data['name'].unique().tolist()
selected_song = st.selectbox('Select a song:', song_names)

# Display recommendations when a song is selected
if selected_song:
    song_list = [{'name': selected_song}]
    recommended_songs = recommend_songs_ui(song_list, data)
    st.write(f"Recommended songs for {selected_song}:")
    for _, row in recommended_songs.iterrows():
        st.write(row['name'])
