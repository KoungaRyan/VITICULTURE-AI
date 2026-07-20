"""
Image Agent — agent spécialisé diagnostic par similarité vectorielle image.

Client MCP connecté UNIQUEMENT à mcp_servers/image_rag_server.py.
"""
from __future__ import annotations
import asyncio
import os

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

SYSTEM_IMAGE = """Tu es l'agent Image RAG de VITI-AI.
Tu réponds UNIQUEMENT sur le diagnostic d'une maladie de la vigne à partir
d'une photo, par comparaison vectorielle avec les images de référence déjà
labellisées dans le Data Commons. N'invente jamais de diagnostic sans avoir
appelé ton tool sur une image réelle (chemin/URL fourni). Si la confiance
renvoyée est faible, dis-le clairement plutôt que de trancher. Réponds en
français, sois concis (max 120 mots)."""

_mcp_client = MultiServerMCPClient(
    {
        "image_rag": {
            "command": "python",
            "args": [os.path.join(os.path.dirname(__file__), "..", "mcp_servers", "image_rag_server.py")],
            "transport": "stdio",
        }
    }
)


async def _load_tools():
    return await _mcp_client.get_tools(server_name="image_rag")


def build_image_agent(model_name: str = OLLAMA_MODEL):
    """Construit le sous-agent React connecté au serveur MCP RAG image."""
    tools = asyncio.run(_load_tools())
    model = ChatOllama(model=model_name, temperature=0.1)
    return create_react_agent(model, tools, prompt=SYSTEM_IMAGE)


# Instance prête à l'emploi, importée par le Supervisor.
image_agent = build_image_agent()
