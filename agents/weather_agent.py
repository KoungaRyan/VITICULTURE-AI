"""
Weather Agent — agent spécialisé météo/agronomie.

Client MCP connecté UNIQUEMENT à mcp_servers/weather_server.py. Cet agent
ne voit ni les tools RAG image, ni les tools RAG texte : son prompt et ses
capacités sont restreints à son domaine.
"""
from __future__ import annotations
import asyncio
import os

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

SYSTEM_WEATHER = """Tu es l'agent Météo/Agronomie de VITI-AI.
Tu réponds UNIQUEMENT sur : conditions météo, risques phytosanitaires
(mildiou/oïdium/botrytis), seuils d'alerte officiels par maladie/stade, et
calculs agronomiques (degrés-jours, dose de traitement, potentiel alcool,
indice EPI mildiou, surface foliaire). Utilise systématiquement tes tools
plutôt que d'estimer à l'œil. Réponds en français, sois concis (max 120 mots),
cite les chiffres exacts renvoyés par tes tools."""

_mcp_client = MultiServerMCPClient(
    {
        "weather": {
            "command": "python",
            "args": [os.path.join(os.path.dirname(__file__), "..", "mcp_servers", "weather_server.py")],
            "transport": "stdio",
        }
    }
)


async def _load_tools():
    return await _mcp_client.get_tools(server_name="weather")


def build_weather_agent(model_name: str = OLLAMA_MODEL):
    """Construit le sous-agent React connecté au serveur MCP météo."""
    tools = asyncio.run(_load_tools())
    model = ChatOllama(model=model_name, temperature=0.1)
    return create_react_agent(model, tools, prompt=SYSTEM_WEATHER)


# Instance prête à l'emploi, importée par le Supervisor.
weather_agent = build_weather_agent()
