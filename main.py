import logging
import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()


class Config:
    DATA_SOURCE = os.getenv("DATA_SOURCE", "movies.csv")
    LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO")


# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOGGING_LEVEL.upper()))


def clean_text(text: str) -> str:
    """
    Cleans text by handling NaN values and removing extra spaces.

    Args:
        text (str): Input text that may contain unwanted spaces or be NaN.

    Returns:
        str: Cleaned text with spaces stripped, or an empty string if NaN.
    """
    if pd.isna(text):
        return ''
    return ' '.join(str(text).split())


class Movie:
    """
    Represents a movie object with various attributes such as title, genres, etc.

    Attributes:
        title (str): Movie title.
        genres (str): Genres associated with the movie.
        keywords (str): Keywords describing the movie.
        companies (str): Production companies.
        popularity (float): Popularity score of the movie.
        release_date (str): Movie's release date.
        runtime (float): Runtime of the movie.
        cast (str): Main cast members.
        vote_count (int): Number of votes.
        vote_average (float): Average vote rating.
    """
    
    def __init__(
        self, title: str, genres: str, keywords: str, companies: str,
        popularity: float, release_date: str, runtime: float,
        cast: str, vote_count: int, vote_average: float
    ):
        self.title = title
        self.genres = genres
        self.keywords = keywords
        self.companies = companies
        self.popularity = popularity
        self.release_date = pd.to_datetime(release_date)
        self.release_year = self.release_date.year
        self.runtime = runtime
        self.cast = cast
        self.vote_count = vote_count
        self.vote_average = vote_average

    def __repr__(self):
        return f"Movie('{self.title}')"


class DatabaseInterface:
    """
    Handles loading and processing movie data from a CSV file.

    Attributes:
        data_source (str): Path to the CSV file.
        movies (List[Movie]): A list of Movie objects created from the data.
    """

    def __init__(self, data_source: str = Config.DATA_SOURCE):
        self.data_source = data_source
        self.movies: List[Movie] = []

    def load_data(self) -> List[Movie]:
        """
        Loads movie data from the CSV file and initializes Movie objects.

        Returns:
            List[Movie]: A list of Movie objects created from the CSV file.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            Exception: If any other error occurs during loading or processing.
        """
        if not os.path.exists(self.data_source):
            logging.error(f"File {self.data_source} does not exist.")
            raise FileNotFoundError(f"File {self.data_source} not found.")
        
        try:
            df = pd.read_csv(self.data_source)
        except Exception as e:
            logging.error(f"Error loading CSV: {e}")
            raise

        try:
            self.movies = [
                Movie(
                    title=row['original_title'],
                    genres=clean_text(row['genres']),
                    keywords=clean_text(row['keywords']),
                    companies=clean_text(row['production_companies']),
                    popularity=row['popularity'] if pd.notna(row['popularity']) else 0.0,
                    release_date=row['release_date'] if pd.notna(row['release_date']) else '1900-01-01',
                    runtime=row['runtime'] if pd.notna(row['runtime']) else 0.0,
                    cast=clean_text(row['cast']),
                    vote_count=row['vote_count'] if pd.notna(row['vote_count']) else 0,
                    vote_average=row['vote_average'] if pd.notna(row['vote_average']) else 0.0
                )
                for _, row in df.iterrows()
            ]
            logging.info(f"Successfully loaded {len(self.movies)} movies.")
        except KeyError as e:
            logging.error(f"Missing expected column in data: {e}")
            raise
        except Exception as e:
            logging.error(f"Error processing data: {e}")
            raise

        return self.movies


class MovieRecommender:
    """
    Movie recommender system that calculates similarity between movies.

    Attributes:
        movies (List[Movie]): List of Movie objects.
        feature_matrix (Optional[np.ndarray]): Matrix of movie features.
        movie_indices (dict): Dictionary mapping movie titles to their indices.
    """

    def __init__(self, movies: List[Movie]):
        self.movies = movies
        self.feature_matrix: Optional[np.ndarray] = None
        self.movie_indices: dict = {movie.title: i for i, movie in enumerate(self.movies)}

    def feature_engineering(self):
        """
        Creates a feature matrix using TF-IDF for text fields and normalizes
        continuous fields.
        """
        vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split('|'), stop_words='english', token_pattern=None)

        genres_matrix = vectorizer.fit_transform([movie.genres for movie in self.movies])
        keywords_matrix = vectorizer.fit_transform([movie.keywords for movie in self.movies])
        cast_matrix = vectorizer.fit_transform([movie.cast for movie in self.movies])

        combined_features = np.hstack(
            [genres_matrix.toarray(), keywords_matrix.toarray(), cast_matrix.toarray()]
        )

        current_year = datetime.now().year
        scaler = MinMaxScaler()
        continuous_features = scaler.fit_transform(np.array([
            [
                (current_year - movie.release_year),
                movie.runtime,
                movie.popularity,
                movie.vote_average,
                np.log1p(movie.vote_count)
            ]
            for movie in self.movies
        ]))

        self.feature_matrix = np.hstack([combined_features, continuous_features])

    def calculate_similarity(self, target_movie_title: str) -> List[str]:
        """
        Calculates the similarity of the target movie with other movies.

        Args:
            target_movie_title (str): The title of the movie to compare.

        Returns:
            List[str]: List of recommended movie titles.

        Raises:
            ValueError: If the movie is not found in the database.
        """
        if self.feature_matrix is None:
            self.feature_engineering()

        target_idx = self.movie_indices.get(target_movie_title)
        if target_idx is None:
            logging.error(f"Movie '{target_movie_title}' not found.")
            raise ValueError(f"Movie '{target_movie_title}' not found in the database.")

        similarity_scores = cosine_similarity([self.feature_matrix[target_idx]], self.feature_matrix)[0]

        most_similar_indices = similarity_scores.argsort()[::-1][1:6]  # Exclude the target movie itself
        recommendations = [self.movies[i].title for i in most_similar_indices]
        logging.info(f"Recommendations for '{target_movie_title}': {recommendations}")
        return recommendations


class Interface:
    """
    Handles interaction with the user.

    Attributes:
        movie_recommender (MovieRecommender): Instance of the recommender system.
    """

    def __init__(self, movie_recommender: MovieRecommender):
        self.movie_recommender = movie_recommender

    def frontscreen(self):
        """Displays the main menu."""
        print("\nWelcome to the Movie Recommendation System!")
        print("1. Get Movie Recommendations")
        print("2. Exit")

    def recommendation_screen(self, recommendations: List[str]):
        """
        Displays the list of recommended movies.

        Args:
            recommendations (List[str]): List of recommended movie titles.
        """
        print("\nWe recommend the following movies:")
        for i, movie in enumerate(recommendations, 1):
            print(f"{i}. {movie}")

    def exit_screen(self):
        """Displays the exit message."""
        print("\nThank you for using the Movie Recommendation System. Goodbye!")

    def run(self):
        """Runs the main loop of the interface, handling user inputs."""
        while True:
            self.frontscreen()
            try:
                user_choice = int(input("\nPlease enter your choice (1 or 2): "))
                if user_choice not in [1, 2]:
                    raise ValueError
            except ValueError:
                print("Invalid input. Please enter 1 or 2.")
                continue

            if user_choice == 2:
                self.exit_screen()
                break

            movie = input("\nEnter the title of the movie: ")
            try:
                recommendations = self.movie_recommender.calculate_similarity(movie)
                self.recommendation_screen(recommendations)
            except ValueError as e:
                logging.error(f"Error: {e}")
                print(f"Error: {e}")


if __name__ == "__main__":
    db = DatabaseInterface()
    movies = db.load_data()
    recommender = MovieRecommender(movies)
    interface = Interface(recommender)
    interface.run()
