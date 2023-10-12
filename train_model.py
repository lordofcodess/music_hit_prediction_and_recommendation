# # Import necessary libraries
# import pandas as pd
# import pickle
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split

# # Load the data
# data = pd.read_csv('SpotifyFeatures.csv')

# # Preprocessing steps based on the document
# data.loc[data["mode"] == 'Major', "mode"] = 1
# data.loc[data["mode"] == 'Minor', "mode"] = 0

# list_of_keys = data['key'].unique()
# for i in range(len(list_of_keys)):
#     data.loc[data['key'] == list_of_keys[i], 'key'] = i

# # Splitting the data
# X = data[['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'mode', 'speechiness', 'tempo', 'valence']]
# y = data['popularity']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Training the Random Forest model
# model = RandomForestRegressor(n_estimators=50, random_state=42)
# model.fit(X_train, y_train)

# # Save the trained model as a pickle file
# with open('model.pkl', 'wb') as file:
#     pickle.dump(model, file)

# print("Model saved as 'model.pkl'")
# Import necessary libraries
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier  # Change here
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('SpotifyFeatures.csv')

# Preprocessing steps based on the document
data.loc[data["mode"] == 'Major', "mode"] = 1
data.loc[data["mode"] == 'Minor', "mode"] = 0

list_of_keys = data['key'].unique()
for i in range(len(list_of_keys)):
    data.loc[data['key'] == list_of_keys[i], 'key'] = i

# Splitting the data
X = data[['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'mode', 'speechiness', 'tempo', 'valence']]
y = (data['popularity'] >= 60).astype(int)  # Change here: Convert to binary classification

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Random Forest model
model = RandomForestClassifier(n_estimators=40, random_state=42)  # Change here
model.fit(X_train, y_train)

# Save the trained model as a pickle file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as 'model.pkl'")
