"""
Serveur MCP — domaine RAG Texte (base documentaire phytosanitaire).

Expose rechercher_connaissance_phytosanitaire (tools/vigne_tools.py) et
ingerer_document (rag/ingestion_rag.py).

Lancement autonome (test) :
    mcp dev mcp_servers/text_rag_server.py
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
from tools.vigne_tools import rechercher_connaissance_phytosanitaire
from rag.ingestion_rag import ingerer_document

mcp = FastMCP("viti-ai-text-rag")


@mcp.tool()
def rechercher_connaissance(question: str, categorie: str | None = None, top_k: int = 3) -> dict:
    """
    Recherche dans la base documentaire phytosanitaire (guides de
    traitement, fiches maladies, réglementation) pour répondre à une
    question technique ou justifier une recommandation.

    Args:
        question: la question ou le sujet à rechercher
        categorie: filtre optionnel ("maladie", "traitement", "reglementation")
        top_k: nombre d'extraits à retourner (défaut 3)
    """
    return rechercher_connaissance_phytosanitaire.func(
        question=question, categorie=categorie, top_k=top_k
    )


@mcp.tool()
def ingerer_doc(texte: str, source: str, categorie: str) -> str:
    """
    Ingère un document texte complet (déjà extrait, ex: via le skill pdf
    pour un PDF de guide phytosanitaire) dans la base RAG.

    Args:
        texte: contenu textuel complet du document
        source: nom/référence du document (ex: "guide_mildiou_2025.pdf")
        categorie: "maladie" | "traitement" | "reglementation"
    """
    n = ingerer_document(texte, source, categorie)
    return f"{n} chunks indexés depuis '{source}'."


if __name__ == "__main__":
    mcp.run(transport="stdio")
