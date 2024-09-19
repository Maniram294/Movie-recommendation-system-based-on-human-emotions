## Movie Recommendation System based on Human emotions

This project is an emotion-based movie recommendation system that uses facial expression analysis to detect the user's emotion and recommend movies accordingly. The system combines computer vision techniques, a deep learning model for emotion detection, and a content-based movie recommendation system using machine learning to suggest movies based on the detected emotions.

### Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage with GUI](#usage-with-gui)
- [Usage without GUI](#usage-without-gui)
- [Emotion-to-Genre Mapping](#emotion-to-genre-mapping)


### Features

- **Real-Time Emotion Detection**: Captures facial expressions using a webcam and detects the user's emotion using a deep learning model.
- **Movie Recommendation Based on Emotion**: Recommends movies based on the detected emotion using a content-based filtering approach.
- **GUI for Recommendations**: Provides a graphical interface to display recommended movies.
- **Flexible Emotion-to-Genre Mapping**: Allows customization of movie genres associated with different emotions.
  
### Requirements

- Python 3.6 or higher
- Required libraries: `pandas`, `numpy`, `random`, `opencv-python`, `tensorflow`, `scikit-learn`, `tkinter`
- Haar Cascade file for face detection
- Pre-trained Keras model for emotion detection (`model_file.h5`)
- Movie datasets (`movies_metadata.csv`, `links_small.csv`, `credits.csv`, `keywords.csv`)

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/emotion-based-movie-recommendation.git
   cd emotion-based-movie-recommendation
   ```

2. **Install Required Libraries:**
   Install the required libraries using `pip`:
   ```bash
   pip install pandas numpy opencv-python tensorflow scikit-learn
   ```

3. **Download Haar Cascade for Face Detection:**
   Download the `haarcascade_frontalface_default.xml` file and place it in the specified directory.

4. **Set Up Your Pre-Trained Model:**
   Place your pre-trained Keras model (`model_file.h5`) in the specified path as mentioned in the code.

5. **Add Movie Datasets:**
   Ensure that the movie datasets (`movies_metadata.csv`, `links_small.csv`, `credits.csv`, `keywords.csv`) are in the `Movies/` directory.

### Usage (With GUI)

1. **Run the Script:**
   ```bash
   python final2.py
   ```

2. **Capture Emotion Using Webcam:**
   The system will use your webcam to capture images and detect emotions for 5 seconds. Ensure your face is clearly visible.
   <p align="center">
   <img src="https://github.com/Maniram294/Movie-recommendation-system-based-on-human-emotions/blob/master/Results/Detected_face-HAPPY.jpg" alt="Capturing Images" width="500"/>
   </p>

3. **View Recommended Movies:**
   A GUI window will display the recommended movies based on the detected emotion. You can refresh the recommendations or close the window as needed.
   <p align="center">
   <img src="https://github.com/Maniram294/Movie-recommendation-system-based-on-human-emotions/blob/master/Results/Recommended_movies-HAPPY.jpg" alt="Recommended Movies" width="500"/>
   </p>


4. **Emotion-to-Movie Recommendations:**
   Based on the most frequently detected emotion, the system will suggest relevant movies. The results are displayed in a separate window.

### Usage (Without GUI)

- **Run the Script:**
   ```bash
   python movies.py
   ```

 - **Console-Based Input:** This version prompts users to enter their emotion and the desired number of movie recommendations directly in the console.
 - **Same Core Functionality Without GUI:** Provides the same emotion-based movie recommendation feature as the GUI version but in a simple, text-based format.

### Emotion-to-Genre Mapping

The emotion-to-genre mapping used in this project is as follows:

- **Happy**: Comedy, Romance
- **Sad**: Drama, Documentary
- **Angry**: Action, Thriller
- **Surprise**: Mystery, Sci-Fi
- **Fear**: Horror
- **Disgust**: Drama
- **Neutral**: Any (selects from all available genres)

Feel free to customize this mapping in the code to better suit your preferences or dataset.


---

## References
- **Emotions Detection**: This project uses the emotions dataset from Kaggle. You can access the dataset and additional information here: [Emotions Dataset](https://www.kaggle.com/code/shivambhardwaj0101/emotion-detection-fer-2013/)
- **Movies Recommendation**: This project uses the movies dataset from Kaggle. You can access the dataset and additional information here: [Movies Dataset](https://www.kaggle.com/code/ibtesama/getting-started-with-a-movie-recommendation-system#Credits,-Genres-and-Keywords-Based-Recommender)

---
