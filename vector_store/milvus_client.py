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
import json
import logging
import math

logger = logging.getLogger("vector_store.client")


# "data/milvus_lite.db" en dev, ou "http://localhost:19530" / cluster en prod
MILVUS_URI = os.environ.get("VITI_MILVUS_URI", "data/milvus_lite.db")
FALLBACK_DIR = os.environ.get("VECTOR_FALLBACK_DIR", "data/vector_fallback")

_env_milvus_uri = os.environ.get("MILVUS_URI")
if _env_milvus_uri and not _env_milvus_uri.startswith(("http://", "https://")):
    logger.warning(
        "Variable d'environnement MILVUS_URI='%s' détectée (chemin local, "
        "pas une URL). pymilvus réserve ce nom et exige http[s]://... : "
        "neutralisée pour cette session. Utilisez VITI_MILVUS_URI à la "
        "place pour configurer ce module (voir .env.example).",
        _env_milvus_uri,
    )
    os.environ.pop("MILVUS_URI", None)

EMBEDDING_DIM_IMAGE = 768   # nomic-embed-vision-v1.5
EMBEDDING_DIM_TEXTE = 768   # nomic-embed-text-v1.5 

COLLECTION_IMAGES = "image_maladies"
COLLECTION_TEXTE = "connaissances_rag"

# -- Détection Milvus (serveur distant OU Lite sur OS compatible) -----------
MILVUS_AVAILABLE = False
try:
    from pymilvus import MilvusClient, DataType  # noqa: F401
    if MILVUS_URI.startswith(("http://", "https://")):
        MILVUS_AVAILABLE = True  # serveur distant : toujours tentable
    else:
        import platform
        if platform.system() in ("Linux", "Darwin"):
            MILVUS_AVAILABLE = True
        else:
            logger.warning(
                "Milvus Lite non supporté sur %s avec URI locale '%s'. "
                "Fallback vectoriel local activé. Pour utiliser un vrai "
                "serveur Milvus: MILVUS_URI=http://localhost:19530 "
                "(voir docker-compose.milvus.yml).",
                platform.system(), MILVUS_URI,
            )
except ImportError:
    logger.warning("pymilvus non installé — fallback vectoriel local activé.")


# ---------------------------------------------------------------------------
# Backend Milvus réel (serveur ou Lite sur Linux/macOS)
# ---------------------------------------------------------------------------
class _MilvusBackend:
    def __init__(self, uri: str):
        from pymilvus import MilvusClient, DataType
        self.client = MilvusClient(uri=uri)
        self._DataType = DataType
        self._init_collections()

    def _init_collections(self) -> None:
        DataType = self._DataType
        if not self.client.has_collection(COLLECTION_IMAGES):
            schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM_IMAGE)
            schema.add_field("id_parcelle", DataType.VARCHAR, max_length=64)
            schema.add_field("maladie", DataType.VARCHAR, max_length=128)
            schema.add_field("severite", DataType.INT64)
            schema.add_field("image_ref", DataType.VARCHAR, max_length=512)
            schema.add_field("date_observation", DataType.VARCHAR, max_length=32)
            idx = self.client.prepare_index_params()
            idx.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
            self.client.create_collection(COLLECTION_IMAGES, schema=schema, index_params=idx)

        if not self.client.has_collection(COLLECTION_TEXTE):
            schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM_TEXTE)
            schema.add_field("texte", DataType.VARCHAR, max_length=4000)
            schema.add_field("source", DataType.VARCHAR, max_length=256)
            schema.add_field("categorie", DataType.VARCHAR, max_length=64)
            idx = self.client.prepare_index_params()
            idx.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
            self.client.create_collection(COLLECTION_TEXTE, schema=schema, index_params=idx)

    def add_image_reference(self, vector, id_parcelle, maladie, severite, image_ref, date_observation):
        self.client.insert(COLLECTION_IMAGES, {
            "vector": vector, "id_parcelle": id_parcelle, "maladie": maladie,
            "severite": severite, "image_ref": image_ref, "date_observation": date_observation,
        })

    def search_images_similaires(self, vector, top_k=5):
        res = self.client.search(COLLECTION_IMAGES, data=[vector], limit=top_k,
            output_fields=["id_parcelle", "maladie", "severite", "image_ref", "date_observation"])
        return [{**h["entity"], "score_similarite": h["distance"]} for h in res[0]]

    def add_connaissance(self, vector, texte, source, categorie):
        self.client.insert(COLLECTION_TEXTE, {
            "vector": vector, "texte": texte, "source": source, "categorie": categorie,
        })

    def search_connaissances(self, vector, top_k=5, categorie=None):
        filter_expr = f'categorie == "{categorie}"' if categorie else ""
        res = self.client.search(COLLECTION_TEXTE, data=[vector], limit=top_k, filter=filter_expr,
            output_fields=["texte", "source", "categorie"])
        return [{**h["entity"], "score_similarite": h["distance"]} for h in res[0]]

    def list_collections(self):
        return self.client.list_collections()


# ---------------------------------------------------------------------------
# Backend fallback local (pur Python + JSON, aucune dépendance externe)
# Suffisant en volume de dev/démo (quelques centaines/milliers de vecteurs).
# ---------------------------------------------------------------------------
class _LocalFallbackBackend:
    def __init__(self, base_dir: str = FALLBACK_DIR):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self._images_path = os.path.join(base_dir, "image_maladies.json")
        self._texte_path = os.path.join(base_dir, "connaissances_rag.json")
        self._images = self._load(self._images_path)
        self._texte = self._load(self._texte_path)
        logger.info("Backend vectoriel fallback local initialisé dans %s", base_dir)

    @staticmethod
    def _load(path: str) -> list:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save(self, path: str, data: list) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def add_image_reference(self, vector, id_parcelle, maladie, severite, image_ref, date_observation):
        self._images.append({
            "vector": vector, "id_parcelle": id_parcelle, "maladie": maladie,
            "severite": severite, "image_ref": image_ref, "date_observation": date_observation,
        })
        self._save(self._images_path, self._images)

    def search_images_similaires(self, vector, top_k=5):
        scored = [
            {**{k: v for k, v in rec.items() if k != "vector"},
             "score_similarite": self._cosine(vector, rec["vector"])}
            for rec in self._images
        ]
        return sorted(scored, key=lambda r: r["score_similarite"], reverse=True)[:top_k]

    def add_connaissance(self, vector, texte, source, categorie):
        self._texte.append({"vector": vector, "texte": texte, "source": source, "categorie": categorie})
        self._save(self._texte_path, self._texte)

    def search_connaissances(self, vector, top_k=5, categorie=None):
        pool = self._texte if categorie is None else [r for r in self._texte if r["categorie"] == categorie]
        scored = [
            {**{k: v for k, v in rec.items() if k != "vector"},
             "score_similarite": self._cosine(vector, rec["vector"])}
            for rec in pool
        ]
        return sorted(scored, key=lambda r: r["score_similarite"], reverse=True)[:top_k]

    def list_collections(self):
        return [COLLECTION_IMAGES, COLLECTION_TEXTE]


# ---------------------------------------------------------------------------
# Façade publique — interface inchangée pour les tools
# ---------------------------------------------------------------------------
class VectorStore:
    def __init__(self, uri: str = MILVUS_URI):
        if MILVUS_AVAILABLE:
            try:
                if "://" not in uri:
                    os.makedirs(os.path.dirname(uri) or ".", exist_ok=True)
                self._backend = _MilvusBackend(uri)
                logger.info("VectorStore: backend Milvus connecté (%s)", uri)
                return
            except Exception as e:
                logger.warning("Connexion Milvus échouée (%s) — bascule fallback local.", e)
        self._backend = _LocalFallbackBackend()
        logger.info("VectorStore: backend fallback local actif")

    @property
    def client(self):
        """Accès brut au backend (ex: client.list_collections() dans les scripts de test)."""
        return self._backend

    def add_image_reference(self, *args, **kwargs):
        return self._backend.add_image_reference(*args, **kwargs)

    def search_images_similaires(self, *args, **kwargs):
        return self._backend.search_images_similaires(*args, **kwargs)

    def add_connaissance(self, *args, **kwargs):
        return self._backend.add_connaissance(*args, **kwargs)

    def search_connaissances(self, *args, **kwargs):
        return self._backend.search_connaissances(*args, **kwargs)