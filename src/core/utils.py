import json


def load_json_file(file_path: str) -> dict:
    with open(file_path, "r") as file:
        return json.load(file)


def search_movies(query: str, movies: dict) -> list:
    return [movie for movie in movies if query in movie["title"].lower()]
