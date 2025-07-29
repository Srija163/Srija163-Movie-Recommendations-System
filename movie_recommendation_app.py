import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import re

# Load datasets
@st.cache_data  
def load_data():
    movies = pd.read_csv('movies.csv')  # Use relative path
    ratings = pd.read_csv('ratings.csv')  # Use relative path
    
    # Extract the year from the movie titles and add it as a separate column
    movies['year'] = movies['title'].apply(lambda x: re.search(r'\((\d{4})\)', x).group(1) if re.search(r'\((\d{4})\)', x) else np.nan)
    movies['year'] = pd.to_numeric(movies['year'], errors='coerce')
    
    # Merge movies and ratings to get average ratings
    avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
    movies = movies.merge(avg_ratings, on='movieId', how='left')
    movies.rename(columns={'rating': 'average_rating'}, inplace=True)
    
    return movies, ratings

movies, ratings = load_data()

st.title('Movie Recommendation System')

# Display dataset preview
if st.checkbox('Show movies dataset'):
    st.write(movies)

if st.checkbox('Show ratings dataset'):
    st.write(ratings)

# Prepare the dataset for collaborative filtering
final_dataset = ratings.pivot(index='movieId', columns='userId', values='rating')
final_dataset.fillna(0, inplace=True)

no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')

# Filter movies and users based on votes
final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index, :]
final_dataset = final_dataset.loc[:, no_movies_voted[no_movies_voted > 50].index]

csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)

# Build the KNN model for collaborative filtering
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

# Movie Recommendation function for collaborative filtering
def get_collaborative_recommendation(movie_name):
    n_movies_to_recommend = 20
    
    # Find the movie id based on the title
    movie_list = movies[movies['title'].str.contains(movie_name, case=False, regex=False)]
    
    if len(movie_list) > 0:
        # Get the movieId from the original movies dataset
        movie_id = movie_list.iloc[0]['movieId']
        
        # Ensure the movie_id exists in the final_dataset
        if movie_id in final_dataset['movieId'].values:
            # Get the index of the movie in the filtered dataset (final_dataset)
            movie_idx = final_dataset[final_dataset['movieId'] == movie_id].index[0]
            
            # Find similar movies using KNN
            distances, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=n_movies_to_recommend+1)
            rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
            
            # Create recommendation list
            recommend_frame = []
            for val in rec_movie_indices:
                movie_idx = final_dataset.iloc[val[0]]['movieId']
                idx = movies[movies['movieId'] == movie_idx].index
                recommend_frame.append({'Title': movies.iloc[idx]['title'].values[0], 'Distance': val[1]})
            
            df = pd.DataFrame(recommend_frame, index=range(1, n_movies_to_recommend+1))
            return df
    return "No similar movies found."

# Movie Recommendation function for content-based filtering
def get_content_based_recommendation(selected_movie):
    # Get the selected movie's genre and average rating
    movie_details = movies[movies['title'] == selected_movie]

    if not movie_details.empty:
        genre = movie_details.iloc[0]['genres']
        avg_rating = movie_details.iloc[0]['average_rating']

        # Filter movies based on the same genre and minimum average rating
        recommended_movies = movies[
            (movies['genres'].str.contains(genre, case=False, regex=False)) & 
            (movies['average_rating'] >= avg_rating)
        ]

        # Sort movies based on average rating (highest first)
        recommended_movies = recommended_movies.sort_values(by='average_rating', ascending=False)

        if not recommended_movies.empty:
            return recommended_movies[['title', 'genres', 'year', 'average_rating']].reset_index(drop=True)
    return "No similar movies found based on genre and rating."

# Choose filtering method
filter_method = st.radio('Select Filtering Method:', ['Content-Based', 'Collaborative'])

# Adding filters for sorting movies in dropdown
genres = set([genre for sublist in movies['genres'].str.split('|').tolist() for genre in sublist])
selected_genre = st.selectbox('Filter by Genre (optional):', ['All'] + sorted(genres))

# Year filter for display only
years = sorted(movies['year'].dropna().unique())
selected_year = st.selectbox('Filter by Year (optional):', ['All'] + [int(year) for year in years])

# Movie selection from the filtered dropdown
filtered_movies = movies.copy()

# Filter based on genre
if selected_genre != 'All':
    filtered_movies = filtered_movies[filtered_movies['genres'].str.contains(selected_genre, case=False, regex=False)]

# Filter based on year for dropdown
if selected_year != 'All':
    filtered_movies = filtered_movies[filtered_movies['year'] == int(selected_year)]

# Movie selection dropdown
if not filtered_movies.empty:
    movie_list_dropdown = st.selectbox('Select a movie to get recommendations:', filtered_movies['title'].unique())

    # Get Recommendations button
    if st.button('Get Recommendations'):
        if filter_method == 'Collaborative':
            recommendations = get_collaborative_recommendation(movie_list_dropdown)
        else:
            recommendations = get_content_based_recommendation(movie_list_dropdown)

        st.write(recommendations)
else:
    st.write("No movies available for the selected filters.")
