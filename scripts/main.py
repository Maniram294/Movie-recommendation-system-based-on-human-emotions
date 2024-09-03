from load_data import load_movie_data
from emotion_recognition import detect_emotions
from recommendations import get_movie_recommendations

# Load movie data
load_movie_data()

# Detect emotion
captured_emotion = detect_emotions()

# Get movie recommendations
if captured_emotion:
    recommended_movies = get_movie_recommendations(captured_emotion, 3)
    for movie in recommended_movies:
        print(movie)
