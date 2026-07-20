"""
Serveur MCP — domaine Météo & Agronomie.

Expose les tools déjà écrits dans tools/vigne_tools.py : get_meteo_vigne, get_seuils_alerte, calcul_agronomique.
Ce serveur ne connaît RIEN du RAG texte ni du RAG image — principe de la séparation par domaine.

Lancement autonome (test) :
    mcp dev mcp_servers/weather_server.py
ou :
    python mcp_servers/weather_server.py
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
from tools.vigne_tools import get_meteo_vigne, get_seuils_alerte, calcul_agronomique

mcp = FastMCP("viti-ai-weather")


@mcp.tool()
def meteo_vigne(latitude: float, longitude: float, jours: int = 3) -> str:
    """
    Météo actuelle + prévisions (Open-Meteo) et indices de risque maladies
    (mildiou, oïdium, botrytis) pour une parcelle de vigne.

    Args:
        latitude: latitude de la parcelle (ex: 44.83 pour Bordeaux)
        longitude: longitude de la parcelle (ex: -0.57 pour Bordeaux)
        jours: nombre de jours de prévision (1 à 7)
    """
    return get_meteo_vigne.func(latitude=latitude, longitude=longitude, jours=jours)


@mcp.tool()
def seuils_alerte(maladie: str, stade_vigne: str) -> str:
    """
    Seuils officiels d'alerte et de traitement pour une maladie ou un
    ravageur de la vigne, selon le stade phénologique.

    Args:
        maladie: mildiou, oïdium, botrytis, excoriose, black-rot, cicadelle,
            eudémis, cochylis, acariens, vers_grise
        stade_vigne: débourrement, feuillaison, floraison, nouaison,
            véraison, maturité
    """
    return get_seuils_alerte.func(maladie=maladie, stade_vigne=stade_vigne)


@mcp.tool()
def calcul_agro(operation: str, parametres: dict | None = None) -> str:
    """
    Calculs agronomiques viticoles.

    Args:
        operation: 'degres_jours' | 'dose_traitement' | 'potentiel_alcool'
            | 'indice_mildiou' | 'surface_feuillaire'
        parametres: paramètres spécifiques à l'opération choisie, ex.
            {"surface_ha": 3.5, "dose_L_ha": 2.5} pour 'dose_traitement'
    """
    return calcul_agronomique.func(operation, **(parametres or {}))


if __name__ == "__main__":
    mcp.run(transport="stdio")
