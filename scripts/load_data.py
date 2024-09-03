import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def load_movie_data():
    global smd  # Declare smd as a global variable
    try:
        # Load metadata
        md = pd.read_csv('Movies/movies_metadata.csv', low_memory=False)
        # Perform data cleaning and preprocessing
        # ...
    except Exception as e:
        print(f"Error loading data: {e}")
        smd = None
