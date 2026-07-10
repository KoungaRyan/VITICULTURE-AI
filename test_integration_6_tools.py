"""
Le script avance par paliers : si un palier échoue, il s'arrête avec un
diagnostic ciblé plutôt que de planter sur une trace Python brute. Chaque
palier peut aussi être commenté/décommenté pour isoler un problème.
"""

from __future__ import annotations
import os
import sys
import time
import traceback

os.environ.setdefault("USE_OLLAMA_EMBEDDINGS", "true")
os.environ.setdefault("MILVUS_URI", "http://localhost:19530")

PALIER_OK = "✅"
PALIER_KO = "❌"
PALIER_WARN = "⚠️ "


def palier(nom):
    def decorateur(fn):
        def wrapper(*args, **kwargs):
            print(f"\n--- {nom} ---")
            t0 = time.time()
            try:
                resultat = fn(*args, **kwargs)
                print(f"{PALIER_OK} OK ({time.time()-t0:.1f}s)")
                return resultat
            except Exception as e:
                print(f"{PALIER_KO} ÉCHEC : {type(e).__name__}: {e}")
                traceback.print_exc(limit=3)
                return None
        return wrapper
    return decorateur


@palier("1. Imports des 6 tools")
def test_imports():
    from tools.vigne_tools import VIGNE_TOOLS
    tools = VIGNE_TOOLS
    noms = [t.name for t in tools]
    print("Tools chargés:", noms)
    assert len(noms) == 6, f"Attendu 6 tools, trouvé {len(noms)}"
    return tools


@palier("2. Data Commons SQLite accessible")
def test_data_commons():
    from data_commons.storage import DataCommonsStore
    store = DataCommonsStore()
    ctx = store.get_historique_parcelle("irouleguy-pilote-01")
    print("Contexte parcelle pilote:", {k: len(v) for k, v in ctx.items()})
    return store


@palier("3. Milvus Lite accessible (images + texte)")
def test_milvus():
    from vector_store.milvus_client import VectorStore
    vs = VectorStore()
    print("Collections:", vs.client.list_collections())
    return vs


@palier("4. Ollama répond (LLM + embeddings)")
def test_ollama():
    import requests
    r = requests.get("http://localhost:11434/api/tags", timeout=5)
    r.raise_for_status()
    modeles = [m["name"] for m in r.json().get("models", [])]
    print("Modèles Ollama disponibles:", modeles)
    if not any("qwen2.5" in m or "llama3" in m for m in modeles):
        print(f"{PALIER_WARN}Aucun modèle LLM attendu trouvé (qwen2.5 / llama3.x)")
    if not any("nomic-embed-text" in m for m in modeles):
        print(f"{PALIER_WARN}nomic-embed-text absent — faites: ollama pull nomic-embed-text")
    return modeles


@palier("5. Embedding texte (via Ollama)")
def test_embedding_texte():
    from embeddings.text_embedder import embed_text
    vec = embed_text("test mildiou vigne")
    print("Dimension du vecteur:", len(vec))
    assert len(vec) > 0
    return vec


@palier("6. Compilation du graphe LangGraph (6 tools + checkpointer)")
def test_graphe():
    from agent.vigne_agent import build_vigne_agent  # adaptez le nom si différent
    graph = build_vigne_agent()
    print("Graphe compilé:", graph)
    return graph


@palier("7. Invocation complète du graphe (question simple, sans image)")
def test_invocation(graph):
    if graph is None:
        print("Palier 6 a échoué, invocation sautée.")
        return None
    result = graph.invoke(
        {"messages": [("user", "Quel est le contexte météo et l'historique de la parcelle irouleguy-pilote-01 ?")]},
        config={"configurable": {"thread_id": "test-integration-6-tools"}},
    )
    dernier_message = result["messages"][-1]
    print("Réponse finale de l'agent:\n", getattr(dernier_message, "content", dernier_message))
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("TEST D'INTÉGRATION — VITI-AI graphe à 6 tools")
    print("=" * 60)

    tools = test_imports()
    store = test_data_commons()
    vs = test_milvus()
    modeles = test_ollama()
    vec = test_embedding_texte()
    graph = test_graphe()
    result = test_invocation(graph)

    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    paliers = {
        "Imports 6 tools": tools is not None,
        "Data Commons": store is not None,
        "Milvus": vs is not None,
        "Ollama": modeles is not None,
        "Embedding texte": vec is not None,
        "Compilation graphe": graph is not None,
        "Invocation complète": result is not None,
    }
    for nom, ok in paliers.items():
        print(f"{PALIER_OK if ok else PALIER_KO} {nom}")

    if not all(paliers.values()):
        print(f"\n{PALIER_WARN}Au moins un palier a échoué — copiez la sortie complète "
              f"pour un diagnostic précis (traces, versions, message d'erreur).")
        sys.exit(1)
    else:
        print("\nTout le pipeline (Data Commons + RAG vectoriel + 6 tools) est opérationnel.")
