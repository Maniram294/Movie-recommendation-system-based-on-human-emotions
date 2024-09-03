import pandas as pd
import numpy as np
import random
import cv2
import time
from ast import literal_eval
from collections import Counter
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import tkinter as tk
from tkinter import ttk

# Load the Haar Cascade for face detection
face_classifier = cv2.CascadeClassifier(
    r'C:\Mine\Projects\Minor Project\harcascades\haarcascade_frontalface_default.xml')

# Load your pre-trained model
model = load_model(r"C:\Mine\Projects\Minor Project\models\model_file.h5")

# Define a dictionary to map the model's output to emotion labels
emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Define the emotion-to-genre mapping
emotion_to_genre = {
    'Happy': ['Comedy', 'Romance'],
    'Sad': ['Drama', 'Documentary'],
    'Angry': ['Action', 'Thriller'],
    'Surprise': ['Mystery', 'Sci-Fi'],
    'Fear': ['Horror'],
    'Disgust': ['Drama'],
    'Neutral': ['Any']
}


# Load and preprocess the movie data
def load_movie_data():
    global smd
    try:
        md = pd.read_csv('Movies/movies_metadata.csv', low_memory=False)
        md = md[md['id'].apply(lambda x: str(x).isdigit())]
        md['id'] = md['id'].astype('int')
        ids_to_drop = [19730, 29503, 35587]
        md = md[~md['id'].isin(ids_to_drop)]
        links_small = pd.read_csv('Movies/links_small.csv')
        links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
        smd = md[md['id'].isin(links_small)]
        if smd.empty:
            print("No movies found after filtering.")
            return
        smd = smd.copy()
        smd['tagline'] = smd['tagline'].fillna('')
        smd['description'] = smd['overview'].fillna('') + smd['tagline']
        smd['description'] = smd['description'].fillna('')

        def safe_literal_eval(val):
            try:
                return literal_eval(val)
            except (ValueError, SyntaxError):
                return []

        smd['genres'] = smd['genres'].fillna('[]').apply(safe_literal_eval)
        smd['genres'] = smd['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.0, stop_words='english')
        tfidf_matrix = tf.fit_transform(smd['description'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        smd = smd.reset_index()
        titles = smd['title']
        indices = pd.Series(smd.index, index=smd['title'])
        credits = pd.read_csv('Movies/credits.csv')
        keywords = pd.read_csv('Movies/keywords.csv')
        keywords['id'] = keywords['id'].astype('int')
        credits['id'] = credits['id'].astype('int')
        md['id'] = md['id'].astype('int')
        md = md.merge(credits, on='id')
        md = md.merge(keywords, on='id')
        smd = md[md['id'].isin(links_small)]
        if smd.empty:
            print("No movies found after merging additional data.")
            return
        print("Movie data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        smd = None


# Function to get movie recommendations based on emotion
def get_movie_recommendations(emotion, num_recommendations):
    if smd is None:
        print("No movie data available.")
        return []
    genres = emotion_to_genre.get(emotion, [])
    if 'Any' in genres:
        genres = list(set(genre for sublist in emotion_to_genre.values() for genre in sublist))
    filtered_movies = smd[smd['genres'].apply(lambda x: any(genre in x for genre in genres))]
    movies_to_recommend = filtered_movies['title'].tolist()
    if not movies_to_recommend:
        print(f"No movies found for genres: {genres}")
    return random.sample(movies_to_recommend, min(len(movies_to_recommend), num_recommendations))


# Function to refresh the movie recommendations
def refresh_recommendations():
    global most_frequent_emotion
    if most_frequent_emotion:
        new_recommendations = get_movie_recommendations(most_frequent_emotion, 3)
        if new_recommendations:
            update_recommendations(new_recommendations)
        else:
            print("No recommendations found for the detected emotion.")
    else:
        print("No emotion detected to refresh recommendations.")


# Function to update the displayed recommendations
def update_recommendations(movies):
    text_widget.delete(1.0, tk.END)
    if not movies:
        text_widget.insert(tk.END, "No recommendations available.")
    for movie in movies:
        text_widget.insert(tk.END, f"- {movie}\n")


# Function to display movie recommendations in a separate window
def display_recommendations(movies,emotion):
    global text_widget
    window = tk.Tk()
    window.title(f"Recommended Movies for {emotion}")

    # Configure window layout
    window.geometry('400x300')
    window.configure(bg='#f0f0f0')

    # Title label
    label = tk.Label(window, text=f"Recommended Movies for {emotion}", font=("Helvetica", 18, 'bold'), bg='#f0f0f0')
    label.pack(pady=10)

    # Create a frame for the recommendations
    frame = tk.Frame(window, bg='#f0f0f0')
    frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

    # Create a scrollbar
    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Create a text widget for displaying recommendations
    text_widget = tk.Text(frame, height=10, width=50, wrap=tk.WORD, font=("Helvetica", 12), bg='#ffffff',
                          yscrollcommand=scrollbar.set)
    text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.config(command=text_widget.yview)

    # Update recommendations
    update_recommendations(movies)

    # Buttons
    button_frame = tk.Frame(window, bg='#f0f0f0')
    button_frame.pack(pady=10, fill=tk.X)

    refresh_button = tk.Button(button_frame, text="Refresh", command=refresh_recommendations, font=("Helvetica", 12),
                               bg='#4CAF50', fg='#ffffff')
    refresh_button.pack(side=tk.LEFT, padx=10)

    close_button = tk.Button(button_frame, text="Close", command=window.destroy, font=("Helvetica", 12), bg='#f44336',
                             fg='#ffffff')
    close_button.pack(side=tk.RIGHT, padx=10)

    window.mainloop()


# Load movie data
load_movie_data()

# Initialize the webcam
cap = cv2.VideoCapture(0)

# List to store captured images and detected emotions
captured_images = []
detected_emotions = []

# Capture 5 images per second for 5 seconds
start_time = time.time()
while time.time() - start_time < 5:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        print("No faces detected")
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        prediction = model.predict(roi)
        emotion_label = emotion_dict[prediction.argmax()]
        captured_images.append(frame.copy())
        detected_emotions.append(emotion_label)
    time.sleep(0.2)

# Release the webcam
cap.release()

if len(detected_emotions) == 0:
    print("No emotions detected.")
else:
    most_frequent_emotion = Counter(detected_emotions).most_common(1)[0][0]
    for idx, emotion in enumerate(detected_emotions):
        if emotion == most_frequent_emotion:
            selected_image = captured_images[idx]
            break
    print("Captured Emotion: ", most_frequent_emotion)
    if smd is not None:
        recommended_movies = get_movie_recommendations(most_frequent_emotion, 3)
        if recommended_movies:
            display_recommendations(recommended_movies,most_frequent_emotion)
        else:
            print("No recommendations found for the detected emotion.")
    else:
        print("Movie data not loaded properly.")
    cv2.putText(selected_image, most_frequent_emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                cv2.LINE_AA)
    cv2.imshow("Detected Emotion", selected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
