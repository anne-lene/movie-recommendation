from main import Movie

import pytest
import pandas as pd 
from datetime import datetime

def test_movie_creation():
    movie = Movie(
        title="Inception", genres="Action|Adventure", keywords="Dream|Spy", 
        companies="Warner Bros", popularity=9.8, release_date="2010-07-16", 
        runtime=148, cast="Leonardo DiCaprio", vote_count=20000, vote_average=8.8
    )

    assert movie.title == "Inception"
    assert movie.genres == "Action|Adventure"
    assert movie.release_year == 2010
    assert movie.runtime == 148
    assert movie.cast == "Leonardo DiCaprio"
    assert isinstance(movie.release_date, pd.Timestamp)
    assert movie.vote_count == 20000
    assert movie.vote_average == 8.8

    # Test the __repr__ method
    assert repr(movie) == "Movie('Inception')"


def test_movie_invalid_date():
    with pytest.raises(ValueError):
        Movie(
            title="Test", genres="Drama", keywords="Keyword",
            companies="Test Company", popularity=5.0, 
            release_date="invalid_date", runtime=120, cast="Actor", 
            vote_count=100, vote_average=7.5
        )