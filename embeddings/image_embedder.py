"""
Wrapper pour nomic-embed-vision-v1.5 (embedding d'images pour l'Image Agent).

nomic-embed-vision-v1.5 partage le même espace vectoriel que
nomic-embed-text-v1.5 : une image et sa description textuelle finissent
proches dans l'espace, ce qui permet une recherche croisée texte<->image
si besoin plus tard (pas utilisé dans cette V1, mais bon à savoir).
"""
from __future__ import annotations
import logging
from functools import lru_cache

logger = logging.getLogger("embeddings.image_embedder")

MODEL_NAME = "nomic-ai/nomic-embed-vision-v1.5"


@lru_cache(maxsize=1)
def _load_model():
    """Chargement paresseux et mis en cache (évite de recharger à chaque appel)."""
    import torch
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    logger.info("Modèle %s chargé sur %s", MODEL_NAME, device)
    return processor, model, device


def embed_image(image_path: str) -> list[float]:
    """Encode une image (chemin local ou URL) en vecteur 768-d normalisé."""
    import torch
    from PIL import Image

    processor, model, device = _load_model()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0]  # token [CLS]
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

    return embedding[0].cpu().tolist()


def embed_images_batch(image_paths: list[str]) -> list[list[float]]:
    """Version batch pour l'ingestion de références (plus rapide qu'un appel par image)."""
    return [embed_image(p) for p in image_paths]
