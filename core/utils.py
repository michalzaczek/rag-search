import json
import string
from nltk.stem import PorterStemmer

with open("data/stopwords.txt", "r") as file:
    STOP_WORDS = set(file.read().splitlines())


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
    return [token for token in tokens if token not in STOP_WORDS]


def stem_tokens(tokens: list) -> list:
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def clean_text(text: str) -> list:
    text = remove_punctuation(text)
    text = text.lower()
    tokens = tokenize(text)
    tokens = remove_stop_words(tokens)
    tokens = stem_tokens(tokens)
    return tokens


def search_movies(query: str, movies: list[dict]) -> list[dict]:
    query_tokens = clean_text(query)
    # check if query is a list of tokens
    titles: list[dict] = []
    for movie in movies:
        movie_title_tokens = clean_text(movie["title"])
        # check if at least one token from the query matches any part of a token from the title
        # if set(query_tokens).intersection(set(movie_title_tokens)):
        #     titles.append(movie)
        for qt in query_tokens:
            if any(qt in token for token in movie_title_tokens):
                titles.append(movie)
                break
    return titles
