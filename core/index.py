import os
from core.utils import clean_text
import pickle


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}

    def __add_document(self, doc_id, text):
        text = clean_text(text)
        for token in text:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term):
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
