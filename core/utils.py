import json
import re
import string


def load_json_file(file_path: str) -> dict:
    with open(file_path, "r") as file:
        return json.load(file)


def remove_punctuation(text: str) -> str:
    punct_list = string.punctuation
    table = str.maketrans("", "", punct_list)
    return text.translate(table)


def tokenize(text: str) -> list:
    # tokenize text and remove empty tokens
    tokens = text.split()
    return [token for token in tokens if token.strip()]


def remove_stop_words(tokens: list) -> list:
    with open("data/stopwords.txt", "r") as file:
        stop_words = file.read().splitlines()
    return [token for token in tokens if token not in stop_words]


def clean_text(text: str) -> list:
    text = remove_punctuation(text)
    text = text.lower()
    tokens = tokenize(text)
    tokens = remove_stop_words(tokens)
    return tokens


def search_movies(query: str, movies: dict) -> list:
    query = clean_text(query)
    # check if query is a list of tokens
    titles = []
    for movie in movies:
        movie_title = clean_text(movie["title"])
        # check if at least one token from the query matches any part of a token from the title
        for query_token in query:
            if any(query_token in token for token in movie_title):
                titles.append(movie)
                break
    return titles
