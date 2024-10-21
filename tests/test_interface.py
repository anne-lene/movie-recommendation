import pytest
from unittest.mock import patch
from main import Movie, MovieRecommender, Interface


@patch("builtins.input", side_effect=["1", "Inception", "2"])
@patch("builtins.print")
def test_interface_run(mock_print, mock_input):
    movies = [
        Movie("Inception", "Action|Adventure", "Dream|Spy", "Warner Bros", 9.8, "2010-07-16", 148, "Leonardo DiCaprio", 20000, 8.8),
        Movie("The Matrix", "Action|Sci-Fi", "Hacker|Future", "Warner Bros", 9.5, "1999-03-31", 136, "Keanu Reeves", 15000, 8.7),
    ]
    recommender = MovieRecommender(movies)
    interface = Interface(recommender)

    interface.run()

    assert mock_print.call_count >= 2  # The number of times the print function was called


@patch("builtins.input", side_effect=["2"])
@patch("builtins.print")
def test_exit(mock_print, mock_input):
    movies = [
        Movie("Inception", "Action|Adventure", "Dream|Spy", "Warner Bros", 9.8, "2010-07-16", 148, "Leonardo DiCaprio", 20000, 8.8)
    ]
    recommender = MovieRecommender(movies)
    interface = Interface(recommender)

    interface.run()

    mock_print.assert_called_with("\nThank you for using the Movie Recommendation System. Goodbye!")
