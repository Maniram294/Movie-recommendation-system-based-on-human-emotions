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
from PIL import Image, ImageTk

# Load the Haar Cascade for face detection
face_classifier = cv2.CascadeClassifier(r'C:\Technical\AntiGravity\Movie-recommendation-system-based-on-human-emotions\harcascades\haarcascade_frontalface_default.xml')

# Load your pre-trained model
model = load_model(r"C:\Technical\AntiGravity\Movie-recommendation-system-based-on-human-emotions\models\model_file.h5")

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

smd = None
cap = None
current_frame = None
captured_emotion = None
live_emotion = None
is_streaming = True

# Style Constants
BG_COLOR = "#0f0f12"
CARD_COLOR = "#18181c"
TEXT_COLOR = "#f1f1f5"
TEXT_MUTED = "#8e8e9f"
ACCENT_COLOR = "#00b4d8"
ACCENT_HOVER = "#0096b4"
SUCCESS_COLOR = "#06d6a0"
SUCCESS_HOVER = "#05b587"
WARNING_COLOR = "#ffb703"
WARNING_HOVER = "#e0a102"
DANGER_COLOR = "#ef476f"
DANGER_HOVER = "#d63c5e"
FONT_FAMILY = "Segoe UI"

# Load and preprocess the movie data
def load_movie_data():
    global smd
    try:
        md = pd.read_csv('Movies_dataset/movies_metadata.csv', low_memory=False)
        md = md[md['id'].apply(lambda x: str(x).isdigit())]
        md['id'] = md['id'].astype('int')
        ids_to_drop = [19730, 29503, 35587]
        md = md[~md['id'].isin(ids_to_drop)]
        links_small = pd.read_csv('Movies_dataset/links_small.csv')
        links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
        
        # Load credits and keywords
        credits = pd.read_csv('Movies_dataset/credits.csv')
        keywords = pd.read_csv('Movies_dataset/keywords.csv')
        keywords['id'] = keywords['id'].astype('int')
        credits['id'] = credits['id'].astype('int')
        
        # Merge datasets before filtering/processing smd
        md = md.merge(credits, on='id')
        md = md.merge(keywords, on='id')
        
        # Filter down to small subset
        smd = md[md['id'].isin(links_small)]
        if smd.empty:
            print("No movies found after merging and filtering.")
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
        
        # Compute TF-IDF matrix & similarity matrix
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.0, stop_words='english')
        tfidf_matrix = tf.fit_transform(smd['description'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        
        smd = smd.reset_index(drop=True)
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
        return []
    return random.sample(movies_to_recommend, min(len(movies_to_recommend), num_recommendations))


# UI helper for button hover
def make_button_interactive(btn, active_bg, normal_bg):
    btn.bind("<Enter>", lambda e: btn.config(background=active_bg))
    btn.bind("<Leave>", lambda e: btn.config(background=normal_bg))


# Camera loop
def update_camera_feed():
    global current_frame, is_streaming, live_emotion
    if not is_streaming or cap is None:
        return
        
    ret, frame = cap.read()
    if ret:
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        current_frame = frame.copy()
        
        # Live face detection and emotion prediction
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        display_frame = frame.copy()
        
        if len(faces) > 0:
            # Take the largest face
            (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (6, 214, 160), 2)
            
            # Predict emotion on the detected face
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            prediction = model.predict(roi, verbose=0)
            live_emotion = emotion_dict[prediction.argmax()]
            
            # Draw emotion label on frame
            cv2.putText(display_frame, live_emotion, (x, max(35, y - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (6, 214, 160), 3, cv2.LINE_AA)
            
            # Update live emotion label in UI
            live_emotion_label.config(text=f"Live Emotion: {live_emotion}", fg=SUCCESS_COLOR)
        else:
            live_emotion = None
            live_emotion_label.config(text="No face detected", fg=WARNING_COLOR)
        
        # Render frame to canvas
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        pil_img = pil_img.resize((540, 405), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        
        camera_canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        camera_canvas.image = tk_img
        
    # Schedule next frame in 30ms (~33 FPS, balances smoothness vs inference cost)
    root.after(30, update_camera_feed)


# Capture emotion action — uses the live emotion already being tracked
def capture_emotion():
    global is_streaming, captured_emotion, current_frame, live_emotion
    if current_frame is None:
        status_label.config(text="Camera not ready", fg=DANGER_COLOR)
        return
    
    if live_emotion is None:
        status_label.config(text="No face detected — point your face at the camera!", fg=WARNING_COLOR)
        return
    
    is_streaming = False  # Freeze camera feed
    captured_emotion = live_emotion  # Use the emotion already detected in the live feed
    
    status_label.config(text=f"Captured Emotion: {captured_emotion}", fg=SUCCESS_COLOR)
    live_emotion_label.config(text=f"Captured: {captured_emotion}", fg=ACCENT_COLOR)
    
    # The last rendered frame already has the face box and emotion overlay,
    # so we just freeze it by stopping the camera loop.
    
    # Show movie recommendations
    load_and_show_recommendations(captured_emotion)
    
    # Toggle control buttons
    capture_btn.pack_forget()
    retake_btn.pack(side=tk.LEFT, padx=10)


# Resume feed action
def resume_camera_feed():
    global is_streaming, captured_emotion, live_emotion
    is_streaming = True
    captured_emotion = None
    live_emotion = None
    status_label.config(text="Camera live. Click 'Capture Emotion'!", fg=TEXT_COLOR)
    live_emotion_label.config(text="Detecting...", fg=TEXT_MUTED)
    
    retake_btn.pack_forget()
    capture_btn.pack(side=tk.LEFT, padx=10)
    
    rec_listbox.delete(0, tk.END)
    rec_listbox.insert(tk.END, "")
    rec_listbox.insert(tk.END, "  🎬 Capture your emotion to see")
    rec_listbox.insert(tk.END, "     movie recommendations.")
    
    update_camera_feed()


# Load & populate recommendations
def load_and_show_recommendations(emotion):
    rec_listbox.delete(0, tk.END)
    
    if smd is None:
        rec_listbox.insert(tk.END, "  Error: Movie database not loaded.")
        return
        
    rec_listbox.insert(tk.END, f"  Emotion: {emotion}")
    rec_listbox.insert(tk.END, "  ───────────────────────────────")
    rec_listbox.insert(tk.END, "")
    
    movies = get_movie_recommendations(emotion, 5)
    if not movies:
        rec_listbox.insert(tk.END, "  No movie recommendations found.")
        return
        
    for i, movie in enumerate(movies, 1):
        rec_listbox.insert(tk.END, f"  {i}. {movie}")
        rec_listbox.insert(tk.END, "")


# Refresh recommendations
def refresh_recommendations():
    global captured_emotion
    if captured_emotion:
        load_and_show_recommendations(captured_emotion)
    else:
        status_label.config(text="Please capture your emotion first!", fg=WARNING_COLOR)


# Closing cleanup
def on_close():
    global cap
    if cap is not None:
        cap.release()
    root.destroy()


# Preload data
print("Loading movie metadata...")
load_movie_data()

# Initialize Tkinter Application
root = tk.Tk()
root.title("Emotion Movie Recommendation System")
root.geometry("980x620")
root.configure(bg=BG_COLOR)
root.resizable(False, False)
root.protocol("WM_DELETE_WINDOW", on_close)

# Header Section
header_frame = tk.Frame(root, bg=BG_COLOR, pady=10)
header_frame.pack(fill=tk.X, padx=20)

title_lbl = tk.Label(header_frame, text="EMOTION-BASED MOVIE RECOMMENDER", font=(FONT_FAMILY, 18, "bold"), bg=BG_COLOR, fg=ACCENT_COLOR)
title_lbl.pack(anchor=tk.W)

subtitle_lbl = tk.Label(header_frame, text="Capture your facial expression to receive personalized movie suggestions", font=(FONT_FAMILY, 10), bg=BG_COLOR, fg=TEXT_MUTED)
subtitle_lbl.pack(anchor=tk.W, pady=(2, 0))

# Content Split Frame
content_frame = tk.Frame(root, bg=BG_COLOR)
content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(5, 15))

# Left Panel (Webcam Feed Card)
left_card = tk.Frame(content_frame, bg=CARD_COLOR, bd=0, highlightthickness=0)
left_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

left_header = tk.Frame(left_card, bg=CARD_COLOR)
left_header.pack(fill=tk.X, padx=15, pady=10)

left_title = tk.Label(left_header, text="Webcam Scanner", font=(FONT_FAMILY, 12, "bold"), bg=CARD_COLOR, fg=TEXT_COLOR, anchor=tk.W)
left_title.pack(side=tk.LEFT)

live_emotion_label = tk.Label(left_header, text="Detecting...", font=(FONT_FAMILY, 11, "bold"), bg=CARD_COLOR, fg=TEXT_MUTED, anchor=tk.E)
live_emotion_label.pack(side=tk.RIGHT)

camera_canvas = tk.Canvas(left_card, width=540, height=405, bg="#0b0b0d", highlightthickness=0)
camera_canvas.pack(padx=15, pady=(0, 10))

control_bar = tk.Frame(left_card, bg=CARD_COLOR, padx=15, pady=10)
control_bar.pack(fill=tk.X)

# Buttons Frame inside Control Bar
btn_frame = tk.Frame(control_bar, bg=CARD_COLOR)
btn_frame.pack(side=tk.LEFT)

capture_btn = tk.Button(btn_frame, text="Capture Emotion", font=(FONT_FAMILY, 10, "bold"), bg=SUCCESS_COLOR, fg=BG_COLOR, activebackground=SUCCESS_HOVER, activeforeground=BG_COLOR, relief=tk.FLAT, bd=0, padx=15, pady=6, command=capture_emotion)
capture_btn.pack(side=tk.LEFT, padx=(0, 10))
make_button_interactive(capture_btn, SUCCESS_HOVER, SUCCESS_COLOR)

retake_btn = tk.Button(btn_frame, text="Retake Photo", font=(FONT_FAMILY, 10, "bold"), bg=WARNING_COLOR, fg=BG_COLOR, activebackground=WARNING_HOVER, activeforeground=BG_COLOR, relief=tk.FLAT, bd=0, padx=15, pady=6, command=resume_camera_feed)
# hidden initially

status_label = tk.Label(control_bar, text="Camera live. Click 'Capture Emotion'!", font=(FONT_FAMILY, 10), bg=CARD_COLOR, fg=TEXT_COLOR, anchor=tk.W)
status_label.pack(side=tk.LEFT, fill=tk.X, padx=10)

# Right Panel (Recommendations Card)
right_card = tk.Frame(content_frame, bg=CARD_COLOR, width=340, bd=0, highlightthickness=0)
right_card.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
right_card.pack_propagate(False)

right_title = tk.Label(right_card, text="Recommendations", font=(FONT_FAMILY, 12, "bold"), bg=CARD_COLOR, fg=TEXT_COLOR, anchor=tk.W, padx=15, pady=10)
right_title.pack(fill=tk.X)

rec_list_frame = tk.Frame(right_card, bg=CARD_COLOR, padx=15)
rec_list_frame.pack(fill=tk.BOTH, expand=True)

rec_listbox = tk.Listbox(rec_list_frame, bg="#111115", fg=TEXT_COLOR, font=(FONT_FAMILY, 11), bd=0, highlightthickness=0, selectbackground=ACCENT_COLOR, selectforeground=BG_COLOR, activestyle="none")
rec_listbox.pack(fill=tk.BOTH, expand=True)

# Initial message in recommendations list
rec_listbox.insert(tk.END, "")
rec_listbox.insert(tk.END, "  🎬 Capture your emotion to see")
rec_listbox.insert(tk.END, "     movie recommendations.")

# Bottom Button Bar for Recommendations
right_btn_bar = tk.Frame(right_card, bg=CARD_COLOR, padx=15, pady=15)
right_btn_bar.pack(fill=tk.X)

refresh_btn = tk.Button(right_btn_bar, text="Refresh Suggestions", font=(FONT_FAMILY, 10, "bold"), bg=ACCENT_COLOR, fg=BG_COLOR, activebackground=ACCENT_HOVER, activeforeground=BG_COLOR, relief=tk.FLAT, bd=0, padx=12, pady=6, command=refresh_recommendations)
refresh_btn.pack(side=tk.LEFT)
make_button_interactive(refresh_btn, ACCENT_HOVER, ACCENT_COLOR)

quit_btn = tk.Button(right_btn_bar, text="Quit", font=(FONT_FAMILY, 10, "bold"), bg=DANGER_COLOR, fg=BG_COLOR, activebackground=DANGER_HOVER, activeforeground=BG_COLOR, relief=tk.FLAT, bd=0, padx=15, pady=6, command=on_close)
quit_btn.pack(side=tk.RIGHT)
make_button_interactive(quit_btn, DANGER_HOVER, DANGER_COLOR)

# Open Webcam Device
print("Initializing camera device...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    status_label.config(text="Error: Could not access camera.", fg=DANGER_COLOR)
    capture_btn.config(state=tk.DISABLED, bg=TEXT_MUTED)
else:
    # Start loop
    update_camera_feed()

# Start Tkinter Event Loop
root.mainloop()
