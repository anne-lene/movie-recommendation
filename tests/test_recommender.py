import pytest
from main import Movie, MovieRecommender


def test_feature_engineering():
    movies = [
        Movie("Inception", "Action|Adventure", "Dream|Spy", "Warner Bros", 9.8, "2010-07-16", 148, "Leonardo DiCaprio", 20000, 8.8),
        Movie("The Matrix", "Action|Sci-Fi", "Hacker|Future", "Warner Bros", 9.5, "1999-03-31", 136, "Keanu Reeves", 15000, 8.7)
    ]
    recommender = MovieRecommender(movies)
    
    recommender.feature_engineering()
    assert recommender.feature_matrix is not None
    assert recommender.feature_matrix.shape[0] == 2  # Should be 2 rows (for 2 movies)


def test_calculate_similarity():
    movies = [
        Movie("Inception", "Action|Adventure", "Dream|Spy", "Warner Bros", 9.8, "2010-07-16", 148, "Leonardo DiCaprio", 20000, 8.8),
        Movie("The Matrix", "Action|Sci-Fi", "Hacker|Future", "Warner Bros", 9.5, "1999-03-31", 136, "Keanu Reeves", 15000, 8.7),
        Movie("The Godfather", "Crime|Drama", "Mafia|Family", "Paramount Pictures", 9.2, "1972-03-24", 175, "Marlon Brando", 12000, 9.2)
    ]
    recommender = MovieRecommender(movies)
    
    recommendations = recommender.calculate_similarity("Inception")
    assert len(recommendations) == 2
    assert "The Matrix" in recommendations
    assert "The Godfather" in recommendations


def test_movie_not_found():
    movies = [
        Movie("Inception", "Action|Adventure", "Dream|Spy", "Warner Bros", 9.8, "2010-07-16", 148, "Leonardo DiCaprio", 20000, 8.8),
    ]
    recommender = MovieRecommender(movies)

    with pytest.raises(ValueError):
        recommender.calculate_similarity("Nonexistent Movie")
