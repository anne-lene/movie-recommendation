import pytest
from unittest.mock import patch, mock_open
from main import DatabaseInterface, Movie
import pandas as pd

def test_load_data_with_real_csv():
    """Test loading data from the actual CSV file."""
    db = DatabaseInterface(data_source="movies.csv")
    movies = db.load_data()

    assert len(movies) > 0  # Ensure movies are loaded from the actual CSV
    assert isinstance(movies[0], Movie)
    assert movies[0].title == "Avatar"