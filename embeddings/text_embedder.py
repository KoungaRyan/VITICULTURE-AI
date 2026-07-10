"""
Wrapper pour nomic-embed-text-v1.5 (embedding de texte pour le RAG modulaire
sur le DATA COMMONS documentaire : guides phytosanitaires, fiches maladies,
réglementation).

Alternative locale (sans dépendance Hugging Face) : si vous préférez rester
100% Ollama comme pour votre LLM, `nomic-embed-text` est aussi disponible en
tant que modèle Ollama (`ollama pull nomic-embed-text`). Une fonction
`embed_text_ollama` est fournie en repli — plus simple à opérer sur votre
setup Windows existant.
"""
from __future__ import annotations
import logging
import os
from functools import lru_cache

logger = logging.getLogger("embeddings.text_embedder")

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
USE_OLLAMA_EMBEDDINGS = os.environ.get("USE_OLLAMA_EMBEDDINGS", "false").lower() == "true"


@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    logger.info("Modèle %s chargé", MODEL_NAME)
    return model


def embed_text_hf(text: str, task_prefix: str = "search_document") -> list[float]:
    """
    task_prefix: "search_document" pour l'indexation, "search_query" pour
    une question utilisateur — nomic-embed-text attend ce préfixe pour
    des résultats optimaux (asymétrie requête/document).
    """
    model = _load_model()
    embedding = model.encode(f"{task_prefix}: {text}", normalize_embeddings=True)
    return embedding.tolist()


def embed_text_ollama(text: str, base_url: str = "http://localhost:11434") -> list[float]:
    """Repli via Ollama local — cohérent avec votre setup OLLAMA_MODEL existant."""
    import requests
    resp = requests.post(
        f"{base_url}/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def embed_text(text: str, task_prefix: str = "search_document") -> list[float]:
    """Point d'entrée unique — bascule HF / Ollama selon USE_OLLAMA_EMBEDDINGS."""
    if USE_OLLAMA_EMBEDDINGS:
        return embed_text_ollama(text)
    return embed_text_hf(text, task_prefix)
