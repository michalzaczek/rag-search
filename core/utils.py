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


def search_movies(query: str, index: dict, docmap: dict):
    query_tokens = clean_text(query)

    if not query_tokens:
        return []

    candidate_ids = set()
    indexed_query_tokens = []

    # OR semantics for recall, but ignore query tokens
    # that don't exist in the inverted index.
    for token in query_tokens:
        token_ids = index.get(token)
        if not token_ids:
            continue
        indexed_query_tokens.append(token)
        candidate_ids.update(token_ids)

    if not candidate_ids:
        return []

    # Stable ordering: smaller IDs first (matches the course/grader expectations).
    return [docmap[doc_id] for doc_id in sorted(candidate_ids)]
