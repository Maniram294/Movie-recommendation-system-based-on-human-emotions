import random

# Define the emotion-to-genre mapping
emotion_to_genre = {
    'Happy': ['Comedy', 'Romance'],
    'Sad': ['Drama', 'Documentary'],
    'Angry': ['Action', 'Thriller'],
    'Surprise': ['Mystery', 'Sci-Fi'],
    'Fear': ['Horror'],
    'Disgust': ['Drama']
}

def get_movie_recommendations(emotion, num_recommendations):
    if smd is None:
        return []
    genres = emotion_to_genre.get(emotion, [])
    # Generate recommendations based on the detected emotion
    # ...
    return random.sample(movies_to_recommend, min(len(movies_to_recommend), num_recommendations))
