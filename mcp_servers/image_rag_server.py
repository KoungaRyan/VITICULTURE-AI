"""
Serveur MCP — domaine RAG Image (diagnostic par similarité vectorielle).

Ré-expose diagnostiquer_image_maladie (tools/vigne_tools.py) et
ingerer_image_reference (image_agent/diagnostic_image_tool.py). Ce serveur
ne connaît RIEN de la météo ni du RAG texte.

Lancement autonome (test) :
    mcp dev mcp_servers/image_rag_server.py
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
from tools.vigne_tools import diagnostiquer_image_maladie
from image_agent.diagnostic_image_tool import ingerer_image_reference

mcp = FastMCP("viti-ai-image-rag")


@mcp.tool()
def diagnostiquer_image(image_path: str, id_parcelle: str, top_k: int = 5) -> dict:
    """
    Diagnostique une maladie de la vigne à partir d'une image (photo
    robot/drone) par similarité vectorielle avec les images de référence
    déjà labellisées dans le Data Commons.

    Args:
        image_path: chemin local ou URL de l'image à diagnostiquer
        id_parcelle: parcelle d'où provient l'image (pour tracer le résultat)
        top_k: nombre d'images de référence à comparer (défaut 5)
    """
    return diagnostiquer_image_maladie.func(
        image_path=image_path, id_parcelle=id_parcelle, top_k=top_k
    )


@mcp.tool()
def ingerer_image(
    image_path: str,
    id_parcelle: str,
    maladie: str,
    severite: int,
    date_observation: str,
) -> str:
    """
    Ajoute une image labellisée (dataset initial, ou validation humaine d'un
    diagnostic robot) au Data Commons vectoriel d'images. C'est ce qui rend
    diagnostiquer_image plus fiable au fil du temps.

    Args:
        image_path: chemin local ou URL de l'image
        id_parcelle: parcelle concernée
        maladie: nom de la maladie confirmée
        severite: sévérité observée (entier)
        date_observation: date au format ISO (YYYY-MM-DD)
    """
    ingerer_image_reference(image_path, id_parcelle, maladie, severite, date_observation)
    return f"Image de référence ingérée pour la parcelle {id_parcelle} ({maladie})."


if __name__ == "__main__":
    mcp.run(transport="stdio")
