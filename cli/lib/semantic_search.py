from sentence_transformers import SentenceTransformer


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

    def generate_embedding(self, text):
        if text.trim() == "":
            raise ValueError("Error: Text cannot be empty")

        return self.model.encode(text)
