import os
from sentence_transformers import SentenceTransformer
import numpy as np
import json

from core.constans import CACHE_DIR
from core.utils import load_json_file


EMBEDDINGS_FILENAME = "movie_embeddings.npy"
EMBEDDINGS_FILEPATH = os.path.join(CACHE_DIR, EMBEDDINGS_FILENAME)


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def verify_embeddings():
    ss = SemanticSearch()
    documents = load_json_file("data/movies.json")["movies"]
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"{embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


class SemanticSearch:
    def __init__(self) -> None:
        # Load the model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings: np.ndarray | None = None
        self.documents: list[dict] = []
        self.document_map = {}

    def generate_embedding(self, text):
        if text.strip() == "":
            raise ValueError("Error: Text cannot be empty")

        return self.model.encode(text)

    def build_embeddings(self, documents):
        self.documents = documents
        doc_repr = []

        for doc in documents:
            doc_id = doc["id"]
            self.document_map[doc_id] = doc
            doc_repr.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(doc_repr, show_progress_bar=True)
        np.save(EMBEDDINGS_FILEPATH, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents

        for doc in documents:
            doc_id = doc["id"]
            self.document_map[doc_id] = doc

        if os.path.exists(EMBEDDINGS_FILEPATH):
            self.embeddings = np.load(EMBEDDINGS_FILEPATH)

        if self.embeddings is not None and len(self.embeddings) == len(documents):
            return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        if not self.documents:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )

        sscore_doc_list = []
        embedded_query = self.generate_embedding(query)
        for i in range(len(self.embeddings)):
            sscore = cosine_similarity(embedded_query, self.embeddings[i])
            doc = self.documents[i]
            sscore_doc_list.append((sscore, doc))

        sscore_sorted = sorted(sscore_doc_list, key=lambda x: x[0], reverse=True)
        sscore_limited = sscore_sorted[:limit]
        return [
            {"score": s[0], "title": s[1]["title"], "description": s[1]["description"]}
            for s in sscore_limited
        ]
