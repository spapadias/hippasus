"""
Shared SentenceTransformer model cache.
Single source of truth across all SLM components to avoid
reloading and MPS recompilation between pipeline runs.
"""
from sentence_transformers import SentenceTransformer

_MODEL_CACHE: dict = {}


def get_or_load_model(model_name: str) -> SentenceTransformer:
    if model_name not in _MODEL_CACHE:
        model = SentenceTransformer(model_name)
        # Warm up with realistic batch size and text length to pre-compile MPS shaders
        dummy_texts = [
            "Base table: credit Predict: class employment housing age purpose",
            "Table: orders amount category price description type value status",
            "Table: customers name address phone email country region city",
            "Table: products category price description weight volume units",
            "Table: transactions amount date type status merchant category",
            "Table: accounts balance type status owner created updated",
            "Table: orders amount category price description type value status",
            "Table: customers name address phone email country region city",
            "Table: products category price description weight volume units",
            "Table: transactions amount date type status merchant category",
            "Table: accounts balance type status owner created updated",
            "Table: orders amount category price description type value status",
            "Table: customers name address phone email country region city",
            "Table: products category price description weight volume units",
            "Table: transactions amount date type status merchant category",
            "Table: accounts balance type status owner created updated",
            "Table: orders amount category price description type value status",
            "Table: customers name address phone email country region city",
            "Table: products category price description weight volume units",
            "Table: transactions amount date type status merchant category",
            "Table: accounts balance type status owner created updated",
            "Table: orders amount category price description type value status",
            "Table: customers name address phone email country region city",
            "Table: products category price description weight volume units",
            "Table: transactions amount date type status merchant category",
            "Table: accounts balance type status owner created updated",
        ]
        model.encode(dummy_texts, show_progress_bar=False)
        _MODEL_CACHE[model_name] = model
    return _MODEL_CACHE[model_name]
