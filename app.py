# # Import necessary libraries
# import streamlit as st
# import pandas as pd
# import pickle

# # Load the trained model
# with open('model.pkl', 'rb') as file:
#     model = pickle.load(file)

# # Create a function to get user input
# def get_user_input():
#     features = {
#         "acousticness": st.sidebar.slider('Acousticness', 0.0, 1.0, 0.5),
#         "danceability": st.sidebar.slider('Danceability', 0.0, 1.0, 0.5),
#         "duration_ms": st.sidebar.number_input('Duration (ms)', min_value=1000, max_value=10000000, value=200000),
#         "energy": st.sidebar.slider('Energy', 0.0, 1.0, 0.5),
#         "instrumentalness": st.sidebar.slider('Instrumentalness', 0.0, 1.0, 0.5),
#         "key": st.sidebar.selectbox('Key', list(range(12))),
#         "liveness": st.sidebar.slider('Liveness', 0.0, 1.0, 0.5),
#         "mode": st.sidebar.selectbox('Mode', [0, 1]),
#         "speechiness": st.sidebar.slider('Speechiness', 0.0, 1.0, 0.5),
#         "tempo": st.sidebar.slider('Tempo', 50.0, 200.0, 120.0),
#         "valence": st.sidebar.slider('Valence', 0.0, 1.0, 0.5)
#     }
#     return pd.DataFrame([features])

# # Main app
# st.title('Spotify Track Hit Prediction App')
# st.write('Enter the track features in the sidebar and get the hit prediction below.')

# # Get user input
# user_input = get_user_input()

# # Display user input
# st.write('User Input:')
# st.write(user_input)

# # Get and display the prediction
# predicted_popularity = model.predict(user_input)
# st.write('Predicted Popularity:')
# st.write(predicted_popularity[0])
# if predicted_popularity[0] >= 60:
#     st.write('Predicted Outcome: This track is likely to be a hit!')
# else:
#     st.write('Predicted Outcome: This track is unlikely to be a hit.')
# Import necessary libraries
import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create a function to get user input
def get_user_input():
    features = {
        "acousticness": st.sidebar.slider('Acousticness', 0.0, 1.0, 0.5),
        "danceability": st.sidebar.slider('Danceability', 0.0, 1.0, 0.5),
        "duration_ms": st.sidebar.number_input('Duration (ms)', min_value=1000, max_value=10000000, value=200000),
        "energy": st.sidebar.slider('Energy', 0.0, 1.0, 0.5),
        "instrumentalness": st.sidebar.slider('Instrumentalness', 0.0, 1.0, 0.5),
        "key": st.sidebar.selectbox('Key', list(range(12))),
        "liveness": st.sidebar.slider('Liveness', 0.0, 1.0, 0.5),
        "mode": st.sidebar.selectbox('Mode', [0, 1]),
        "speechiness": st.sidebar.slider('Speechiness', 0.0, 1.0, 0.5),
        "tempo": st.sidebar.slider('Tempo', 50.0, 200.0, 120.0),
        "valence": st.sidebar.slider('Valence', 0.0, 1.0, 0.5)
    }
    return pd.DataFrame([features])

# Main app
st.title('Spotify Track Hit Prediction App')
st.write('Enter the track features in the sidebar and get the hit prediction below.')

# Get user input
user_input = get_user_input()

# Display user input
st.write('User Input:')
st.write(user_input)

# Get and display the prediction
predicted_outcome = model.predict(user_input)
st.write('Predicted Outcome:')
if predicted_outcome[0] == 1:
    st.write('This track is likely to be a hit!')
else:
    st.write('This track is unlikely to be a hit.')
