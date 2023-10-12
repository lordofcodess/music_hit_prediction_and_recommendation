# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from joblib import dump

# # Load data
# data = pd.read_csv("data.csv")
# genre_data = pd.read_csv('data_by_genres.csv')

# # Check columns in genre_data
# print("Columns in genre_data:", genre_data.columns.tolist())

# # Define the columns (remove 'year' and 'explicit' if they are not present in genre_data)
# number_cols = ['valence', 'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

# # Clustering logic
# X = genre_data[number_cols]
# cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=12))])
# cluster_pipeline.fit(X)

# # Save the trained model
# dump(cluster_pipeline, 'cluster_model.joblib')


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump

# Load data
data = pd.read_csv("data.csv")  # Assuming your data file is named "data.csv"

# Define the numerical columns used for clustering
number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit','instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

# Clustering songs into 25 clusters
song_cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=25, verbose=False))
])

X_song = data[number_cols]
song_cluster_pipeline.fit(X_song)
data['cluster_label'] = song_cluster_pipeline.predict(X_song)

# Save the trained model
dump(song_cluster_pipeline, 'song_cluster_model.joblib')
