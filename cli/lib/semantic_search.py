from sentence_transformers import SentenceTransformer
import numpy as np


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
        self.embeddings = None
        self.documents = None
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

        embeddings = self.model.encode(doc_repr, show_progress_bar=True)
        np.save("cache/movie_embeddings.npy", embeddings)
