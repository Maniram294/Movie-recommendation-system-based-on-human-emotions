import pandas as pd
import random

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


# Define the function to update recommendations
def refresh_recommendations(emotion, num_recommendations):
    genres = emotion_to_genre.get(emotion, [])
    if 'Any' in genres:
        genres = list(set(genre for sublist in emotion_to_genre.values() for genre in sublist))

    # Filter movies based on genres
    filtered_movies = smd[smd['genres'].apply(lambda x: any(genre in x for genre in genres))]

    if emotion == 'Neutral':
        movies_to_recommend = random.sample(all_movies, min(len(all_movies), num_recommendations))
    else:
        movies_to_recommend = filtered_movies['title'].tolist()
        movies_to_recommend = random.sample(movies_to_recommend, min(len(movies_to_recommend), num_recommendations))

    # Create a DataFrame to display
    recommendations_df = pd.DataFrame({
        'Title': movies_to_recommend
    })

    # Print the DataFrame
    print(recommendations_df)


# Load movie data
def load_data():
    try:
        # Load metadata with dtype and low_memory options
        md = pd.read_csv('Movies/movies_metadata.csv', low_memory=False)

        # Display the first few rows to inspect data
        print("First few rows of metadata:")
        print(md.head())
        print("Data types in metadata:")
        print(md.dtypes)

        # Clean and convert the 'id' column
        md = md[md['id'].apply(lambda x: str(x).isdigit())]  # Keep only rows where 'id' is numeric
        md['id'] = md['id'].astype('int')

        # Check for and handle non-integer values in 'id'
        links_small = pd.read_csv('Movies/links_small.csv')
        links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

        # Drop rows with specific IDs
        md = md[~md['id'].isin([19730, 29503, 35587])]

        # Filter movies based on IDs in links_small
        smd = md[md['id'].isin(links_small)]
        print("smd : ",smd)
        # Fill missing values and create 'description'
        smd.loc[:, 'tagline'] = smd['tagline'].fillna('')
        smd.loc[:, 'description'] = smd['overview'].fillna('') + smd['tagline']
        smd.loc[:, 'description'] = smd['description'].fillna('')

        return smd
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


if __name__ == "__main__":
    smd = load_data()

    if smd is not None:
        all_movies = list(smd['title'])

        # Get user input
        emotion = input("Enter emotion (Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral): ").capitalize()
        if emotion not in emotion_to_genre:
            print("Invalid emotion. Please enter one of the valid options.")
        else:
            try:
                num_recommendations = int(input("Enter number of movie recommendations: "))
                if num_recommendations <= 0:
                    raise ValueError("Number of recommendations must be positive.")

                # Get recommendations
                refresh_recommendations(emotion, num_recommendations)
            except ValueError as e:
                print(f"Invalid input: {e}")
    else:
        print("Failed to load data.")
