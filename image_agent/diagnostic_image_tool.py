from __future__ import annotations
from vector_store.milvus_client import VectorStore
from embeddings.image_embedder import embed_image

_vector_store = VectorStore()

def ingerer_image_reference(
    image_path: str, id_parcelle: str, maladie: str,
    severite: int, date_observation: str,
) -> None:
    """
    Fonction d'ingestion — à appeler pour alimenter le
    Data Commons vectoriel avec des images déjà labellisées (dataset initial,
    ou validation humaine d'un diagnostic robot). C'est ce qui rend
    diagnostiquer_image_maladie de plus en plus fiable au fil du temps.
    """
    vecteur = embed_image(image_path)
    _vector_store.add_image_reference(
        vector=vecteur, id_parcelle=id_parcelle, maladie=maladie,
        severite=severite, image_ref=image_path, date_observation=date_observation,
    )
