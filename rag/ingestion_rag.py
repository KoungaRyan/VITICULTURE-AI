"""
Pipeline d'ingestion du RAG modulaire : découpe des documents phytosanitaires
(txt/markdown extraits de PDF via votre skill pdf, guides, fiches) en chunks,
les vectorise et les indexe dans Milvus.
"""
from __future__ import annotations
import logging
import re

from vector_store.milvus_client import VectorStore
from embeddings.text_embedder import embed_text

logger = logging.getLogger("rag.ingestion")

_vector_store = VectorStore()


def chunk_text(text: str, taille_max: int = 800, chevauchement: int = 100) -> list[str]:
    """
    Découpage simple par paragraphes puis regroupement jusqu'à taille_max
    caractères, avec chevauchement pour ne pas couper le contexte au milieu
    d'une info clé (ex: une posologie à cheval sur deux chunks).
    """
    paragraphes = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, courant = [], ""

    for p in paragraphes:
        if len(courant) + len(p) <= taille_max:
            courant = f"{courant}\n\n{p}".strip()
        else:
            if courant:
                chunks.append(courant)
            courant = (courant[-chevauchement:] + "\n\n" + p) if chevauchement else p
    if courant:
        chunks.append(courant)
    return chunks


def ingerer_document(texte: str, source: str, categorie: str) -> int:
    """
    Ingère un document texte complet (déjà extrait, ex: via le skill pdf
    pour un PDF de guide phytosanitaire) dans la base de connaissances RAG.

    Args:
        texte: contenu textuel complet du document
        source: nom/référence du document (ex: "guide_mildiou_2025.pdf")
        categorie: "maladie" | "traitement" | "reglementation"

    Returns:
        nombre de chunks indexés
    """
    chunks = chunk_text(texte)
    for chunk in chunks:
        vecteur = embed_text(chunk, task_prefix="search_document")
        _vector_store.add_connaissance(vecteur, texte=chunk, source=source, categorie=categorie)
    logger.info("Document '%s' ingéré: %d chunks", source, len(chunks))
    return len(chunks)


if __name__ == "__main__":
    # Exemple : ingestion d'une fiche mildiou (à remplacer par vos vrais docs,
    # potentiellement extraits de PDF via le skill pdf de Claude en amont)
    fiche_exemple = """
Le mildiou de la vigne (Plasmopara viticola) se développe principalement par
temps chaud et humide, avec des pluies supérieures à 10mm et des températures
above 10°C. Les premiers symptômes apparaissent sur les feuilles sous forme
de taches jaunâtres translucides ("taches d'huile").

Traitement préventif recommandé : bouillie bordelaise (cuivre) à 6-8 kg/ha
avant les périodes à risque, renouvelée après chaque pluie lessivante
supérieure à 20mm. Délai avant récolte : 21 jours minimum selon le produit.

En cas de forte pression, un traitement systémique peut être nécessaire,
avec un délai avant récolte variable selon la matière active (voir AMM du
produit spécifique).
"""
    n = ingerer_document(fiche_exemple, source="fiche_mildiou_exemple.txt", categorie="maladie")
    print(f"{n} chunks ingérés")
