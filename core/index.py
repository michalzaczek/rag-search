from collections import Counter
import math
import os
from core.utils import clean_text
import pickle
from typing import Any


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict[str, Any]] = {}
        self.term_frequencies: dict[int, Counter[str]] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = clean_text(text)

        self.term_frequencies[doc_id] = Counter(tokens)

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = clean_text(term)

        if len(tokens) != 1:
            raise ValueError(f"Term '{term}' must result in exactly one token.")

        target_token = tokens[0]

        if doc_id in self.term_frequencies:
            return self.term_frequencies[doc_id].get(target_token, 0)

        return 0

    def get_idf(self, term: str) -> float:
        tokens = clean_text(term)

        if len(tokens) != 1:
            raise ValueError(f"Term '{term}' must result in exactly one token.")

        target_token = tokens[0]
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index.get(target_token, set()))
        return math.log(
            (total_doc_count - term_match_doc_count + 0.5)
            / (term_match_doc_count + 0.5)
            + 1
        )

    def get_bm25_idf(self, term: str) -> float:
        # bm25 uses more stable formula
        # IDF = log((N - df + 0.5) / (df + 0.5) + 1)

        tokens = clean_text(term)

        if len(tokens) != 1:
            raise ValueError(f"Term '{term}' must result in exactly one token.")

        target_token = tokens[0]
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index.get(target_token, set()))
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_tfidf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def get_documents(self, term: str) -> list[int]:
        tokens = clean_text(term)
        if not tokens:
            return []
        token = tokens[0]
        return sorted(list(self.index.get(token, set())))

    def build(self, movies: list[dict]):
        for movie in movies:
            doc_id = movie["id"]
            title = movie["title"]
            description = movie["description"]
            combined_text = f"{title} {description}"
            self.__add_document(doc_id, combined_text)
            self.docmap[doc_id] = movie

    def save(self):
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as file:
            pickle.dump(self.index, file)
        with open("cache/docmap.pkl", "wb") as file:
            pickle.dump(self.docmap, file)
        with open("cache/term_frequencies.pkl", "wb") as file:
            pickle.dump(self.term_frequencies, file)

    def load(self):
        try:
            with open("cache/index.pkl", "rb") as file:
                self.index = pickle.load(file)
        except FileNotFoundError:
            self.index = {}

        try:
            with open("cache/docmap.pkl", "rb") as file:
                self.docmap = pickle.load(file)
        except FileNotFoundError:
            self.docmap = {}

        try:
            with open("cache/term_frequencies.pkl", "rb") as file:
                self.term_frequencies = pickle.load(file)
        except FileNotFoundError:
            self.term_frequencies = {}
