"""
Text-RAG Agent — agent spécialisé recherche documentaire phytosanitaire.

Client MCP connecté UNIQUEMENT à mcp_servers/text_rag_server.py.
"""
from __future__ import annotations
import asyncio
import os

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

SYSTEM_TEXT_RAG = """Tu es l'agent RAG Texte de VITI-AI.
Tu réponds UNIQUEMENT en t'appuyant sur la base documentaire phytosanitaire
(guides de traitement, fiches maladies, réglementation) via ton tool de
recherche vectorielle. Cite toujours la source des extraits utilisés. Si
aucun extrait pertinent n'est trouvé, dis-le plutôt que d'inventer une
réglementation ou une posologie. Réponds en français, sois concis (max 120
mots)."""

_mcp_client = MultiServerMCPClient(
    {
        "text_rag": {
            "command": "python",
            "args": [os.path.join(os.path.dirname(__file__), "..", "mcp_servers", "text_rag_server.py")],
            "transport": "stdio",
        }
    }
)


async def _load_tools():
    return await _mcp_client.get_tools(server_name="text_rag")


def build_text_rag_agent(model_name: str = OLLAMA_MODEL):
    """Construit le sous-agent React connecté au serveur MCP RAG texte."""
    tools = asyncio.run(_load_tools())
    model = ChatOllama(model=model_name, temperature=0.1)
    return create_react_agent(model, tools, prompt=SYSTEM_TEXT_RAG)


# Instance prête à l'emploi, importée par le Supervisor.
text_rag_agent = build_text_rag_agent()
