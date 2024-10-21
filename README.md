
# Movie Recommendation System

This project is a **Movie Recommendation System** that uses **cosine similarity** to recommend movies based on different features such as genres, keywords, cast, and more. The project leverages machine learning techniques like **TF-IDF Vectorization** and **cosine similarity** to find similar movies and provides a simple command-line interface for interacting with the system.

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [Testing](#testing)
7. [Technologies Used](#technologies-used)
8. [License](#license)

## Features

- **Movie Recommendation**: Given a movie title, the system recommends similar movies based on features like genres, cast, and keywords.
- **TF-IDF Vectorization**: Converts text features (like genres and keywords) into numerical vectors for comparison.
- **Cosine Similarity**: Measures the similarity between movies based on their feature vectors.
- **Interactive CLI Interface**: Users can get recommendations by entering a movie title directly into the command line.
- **Error Handling**: Graceful error handling for missing or invalid inputs.
- **Configurable**: Uses environment variables and a `.env` file for configuration.

## Project Structure

```
movie-recommendation/
├── main.py                # Contains core logic (Movie, DatabaseInterface, MovieRecommender, Interface)
├── tests/                 # Unit and integration tests
│   ├── test_database.py    # Tests for data loading functionality
│   ├── test_movie.py       # Tests for the Movie class
│   ├── test_interface.py   # Tests for the user interface
│   ├── test_recommender.py # Tests for recommendation logic
├── .env                   # Environment variables
├── pyproject.toml         # Poetry configuration file
├── pytest.ini             # Pytest configuration
└── README.md              # Project documentation (this file)
```

## Installation

### Prerequisites

- **Python 3.8+**
- **Poetry** (for dependency management)

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/anne-lene/movie-recommendation.git
   cd movie-recommendation
   ```

2. Install dependencies using **Poetry**:

   ```bash
   poetry install
   ```

3. Create a `.env` file for environment variables:

   ```bash
   touch .env
   ```

   Add the following content to your `.env` file (you can modify the values as needed):

   ```
   DATA_SOURCE=movies.csv
   LOGGING_LEVEL=INFO
   ```

4. (Optional) If you plan to use real data, place your `movies.csv` file in the project root or configure the path in `.env`.

## Usage

To use the Movie Recommendation System, you can run the program from the command line:

```bash
poetry run python main.py
```

You will be prompted to enter the title of a movie, and the system will recommend similar movies based on the provided data.

### Example:

```
Welcome to the Movie Recommendation System!
1. Get Movie Recommendations
2. Exit

Please enter your choice (1 or 2): 1
Enter the title of the movie: Inception

We recommend the following movies:
1. The Matrix
2. The Dark Knight
3. Interstellar
```

## Configuration

The application uses environment variables for configuration. The default values can be set in the `.env` file.

- **DATA_SOURCE**: Path to the CSV file containing the movie data (e.g., `movies.csv`).
- **LOGGING_LEVEL**: Set the logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`).

## Testing

The project includes unit and integration tests. To run the tests:

1. Ensure that **pytest** is installed:

   ```bash
   poetry add --dev pytest
   ```

2. Run the tests:

   ```bash
   poetry run pytest
   ```

This will run all tests in the `tests/` folder.

### Test Structure:

- **`test_database.py`**: Tests for loading movie data from the CSV file.
- **`test_movie.py`**: Tests for the `Movie` class and its attributes.
- **`test_recommender.py`**: Tests for the `MovieRecommender` class and similarity calculations.
- **`test_interface.py`**: Tests for the command-line interface functionality.

## Technologies Used

- **Python 3.8+**
- **Pandas**: For data manipulation and CSV reading.
- **Scikit-learn**: For `TfidfVectorizer` and `cosine_similarity`.
- **Poetry**: For dependency management.
- **Pytest**: For testing.
- **Python-dotenv**: For managing environment variables.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Data

The initial movie data was sourced from Kaggle's Movies CSV Dataset: [Movies CSV Dataset](https://www.kaggle.com/datasets/harshshinde8/movies-csv).
