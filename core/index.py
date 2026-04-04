from collections import Counter
import math
import os
from statistics import mean
from core.utils import clean_text
import pickle
from typing import Any
from core.constans import BM25_B, BM25_K1, CACHE_DIR


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict[str, Any]] = {}
        self.term_frequencies: dict[int, Counter[str]] = {}
        self.doc_lengths: dict[int, int] = {}
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = clean_text(text)

        self.term_frequencies[doc_id] = Counter(tokens)
        self.doc_lengths[doc_id] = len(tokens)

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        return mean(self.doc_lengths.values())

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
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        # bm25 uses more stable formula
        # IDF = log((N - df + 0.5) / (df + 0.5) + 1)

        tokens = clean_text(term)

        if len(tokens) != 1:
            raise ValueError(f"Term '{term}' must result in exactly one token.")

        target_token = tokens[0]
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index.get(target_token, set()))
        numerator = total_doc_count - term_match_doc_count + 0.5
        denominator = term_match_doc_count + 0.5

        return math.log(numerator / denominator + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1: float, b: float) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        # Length normalization factor
        length_norm = 1 - b + b * (doc_length / avg_doc_length)

        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        tf = self.get_bm25_tf(doc_id, term, k1, b)
        idf = self.get_bm25_idf(term)
        return tf * idf

    def bm25_search(self, query: str, limit: int):
        tokens = clean_text(query)
        scores = {}

        for token in tokens:
            doc_ids = self.index.get(token, set())

            for di in doc_ids:
                scores[di] = scores.get(di, 0) + self.bm25(di, token)

        sorted_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        top_results = sorted_items[:limit]

        final_results = []
        for doc_id, score in top_results:
            final_results.append({"document": self.docmap[doc_id], "score": score})

        return final_results

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
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(os.path.join(CACHE_DIR, "index.pkl"), "wb") as file:
            pickle.dump(self.index, file)
        with open(os.path.join(CACHE_DIR, "docmap.pkl"), "wb") as file:
            pickle.dump(self.docmap, file)
        with open(os.path.join(CACHE_DIR, "term_frequencies.pkl"), "wb") as file:
            pickle.dump(self.term_frequencies, file)
        with open(self.doc_lengths_path, "wb") as file:
            pickle.dump(self.doc_lengths, file)

    def load(self):
        try:
            with open(os.path.join(CACHE_DIR, "index.pkl"), "rb") as file:
                self.index = pickle.load(file)
        except FileNotFoundError:
            self.index = {}

        try:
            with open(os.path.join(CACHE_DIR, "docmap.pkl"), "rb") as file:
                self.docmap = pickle.load(file)
        except FileNotFoundError:
            self.docmap = {}

        try:
            with open(os.path.join(CACHE_DIR, "term_frequencies.pkl"), "rb") as file:
                self.term_frequencies = pickle.load(file)
        except FileNotFoundError:
            self.term_frequencies = {}

        try:
            with open(self.doc_lengths_path, "rb") as file:
                self.doc_lengths = pickle.load(file)
        except FileNotFoundError:
            self.doc_lengths = {}
