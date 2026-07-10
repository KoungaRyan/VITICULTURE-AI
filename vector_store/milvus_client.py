"""
Client Milvus pour VITI-AI — deux collections :
  - image_maladies   : embeddings d'images (nomic-embed-vision-v1.5) pour l'Image Agent
  - connaissances_rag: embeddings de texte (docs phytosanitaires) pour le RAG modulaire

Mode dev (par défaut) : Milvus Lite, base fichier locale, aucun serveur requis
    -> cohérent avec votre pattern SQLite pour le Data Commons / checkpointer.
Mode prod : serveur Milvus distant (Docker / cluster), activé via MILVUS_URI.

Bascule automatique via variable d'environnement, sur le même principe que
SQLITE_AVAILABLE pour SqliteSaver.
"""
from __future__ import annotations
import os
import logging
from pymilvus import MilvusClient, DataType

logger = logging.getLogger("vector_store.milvus_client")

# "data/milvus_lite.db" en dev, ou "http://localhost:19530" / cluster en prod
MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")

EMBEDDING_DIM_IMAGE = 768   # nomic-embed-vision-v1.5
EMBEDDING_DIM_TEXTE = 768   # nomic-embed-text-v1.5 

COLLECTION_IMAGES = "image_maladies"
COLLECTION_TEXTE = "connaissances_rag"


class VectorStore:
    """Point d'accès unique aux collections vectorielles Milvus."""

    def __init__(self, uri: str = MILVUS_URI):
        self.uri = uri
        os.makedirs(os.path.dirname(uri) or ".", exist_ok=True) if "://" not in uri else None
        self.client = MilvusClient(uri=uri)
        self._init_collections()
        logger.info("VectorStore connecté: %s", uri)

    # ------------------------------------------------------------------
    def _init_collections(self) -> None:
        if not self.client.has_collection(COLLECTION_IMAGES):
            schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM_IMAGE)
            schema.add_field("id_parcelle", DataType.VARCHAR, max_length=64)
            schema.add_field("maladie", DataType.VARCHAR, max_length=128)
            schema.add_field("severite", DataType.INT64)
            schema.add_field("image_ref", DataType.VARCHAR, max_length=512)
            schema.add_field("date_observation", DataType.VARCHAR, max_length=32)

            index_params = self.client.prepare_index_params()
            index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")

            self.client.create_collection(
                collection_name=COLLECTION_IMAGES, schema=schema, index_params=index_params
            )
            logger.info("Collection '%s' créée", COLLECTION_IMAGES)

        if not self.client.has_collection(COLLECTION_TEXTE):
            schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM_TEXTE)
            schema.add_field("texte", DataType.VARCHAR, max_length=4000)
            schema.add_field("source", DataType.VARCHAR, max_length=256)
            schema.add_field("categorie", DataType.VARCHAR, max_length=64)  # ex: maladie, traitement, reglementation

            index_params = self.client.prepare_index_params()
            index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")

            self.client.create_collection(
                collection_name=COLLECTION_TEXTE, schema=schema, index_params=index_params
            )
            logger.info("Collection '%s' créée", COLLECTION_TEXTE)

    # -- Image Agent ------------------------------------------------------
    def add_image_reference(
        self, vector: list[float], id_parcelle: str, maladie: str,
        severite: int, image_ref: str, date_observation: str,
    ) -> None:
        self.client.insert(COLLECTION_IMAGES, {
            "vector": vector, "id_parcelle": id_parcelle, "maladie": maladie,
            "severite": severite, "image_ref": image_ref, "date_observation": date_observation,
        })

    def search_images_similaires(self, vector: list[float], top_k: int = 5) -> list[dict]:
        res = self.client.search(
            COLLECTION_IMAGES, data=[vector], limit=top_k,
            output_fields=["id_parcelle", "maladie", "severite", "image_ref", "date_observation"],
        )
        return [
            {**hit["entity"], "score_similarite": hit["distance"]}
            for hit in res[0]
        ]

    # -- RAG texte ----------------------------------------------------------
    def add_connaissance(self, vector: list[float], texte: str, source: str, categorie: str) -> None:
        self.client.insert(COLLECTION_TEXTE, {
            "vector": vector, "texte": texte, "source": source, "categorie": categorie,
        })

    def search_connaissances(self, vector: list[float], top_k: int = 5, categorie: str | None = None) -> list[dict]:
        filter_expr = f'categorie == "{categorie}"' if categorie else ""
        res = self.client.search(
            COLLECTION_TEXTE, data=[vector], limit=top_k, filter=filter_expr,
            output_fields=["texte", "source", "categorie"],
        )
        return [
            {**hit["entity"], "score_similarite": hit["distance"]}
            for hit in res[0]
        ]
