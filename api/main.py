"""
API REST FastAPI — Agent Agronome Viticole VITI-AI
===================================================
v1.2 — SqliteSaver + Historique des conversations
  - SqliteSaver : persistance de l'historique entre redémarrages
  - GET  /historique              → liste toutes les sessions
  - GET  /historique/{thread_id}  → messages complets d'une session
  - DELETE /historique/{thread_id} → supprime une session
  - Timeout configurable via variables d'environnement
  - Gestion fine des erreurs Ollama (wsarecv, timeout, connexion refusée)
"""
import json
import os
import sys
import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as PydanticBaseModel, Field

sys.path.insert(0, '/home/claude/vigne_agent')
from agent.vigne_agent import build_vigne_agent, chat_avec_agent
from tools.vigne_tools import get_meteo_vigne, get_seuils_alerte, calcul_agronomique
from models.schemas import DiagnosticVigne
from data_commons.api_router import router as data_commons_router



# ─────────────────────────────────────────
# CONFIG (modifiable via variables d'env)
# ─────────────────────────────────────────
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL",   "llama3.2:3b")
OLLAMA_URL     = os.getenv("OLLAMA_URL",     "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
OLLAMA_PREDICT = int(os.getenv("OLLAMA_PREDICT", "512"))
USE_OLLAMA     = os.getenv("USE_OLLAMA", "true").lower() == "true"
DB_PATH        = os.getenv("DB_PATH", "data/viti_ai_conversations.db")


# ─────────────────────────────────────────
# MODÈLES PYDANTIC
# ─────────────────────────────────────────
class ChatRequest(PydanticBaseModel):
    message: str = Field(..., description="Message de l'utilisateur")
    thread_id: str = Field(default="session_001", description="ID de session (mémoire)")
    parcelle_context: Optional[Dict[str, Any]] = Field(default=None)

class ChatResponse(PydanticBaseModel):
    response: str
    diagnostic: Optional[Dict] = None
    thread_id: str
    messages_count: int
    timestamp: str

class MeteoRequest(PydanticBaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    jours: int = Field(default=3, ge=1, le=7)

class SeuilRequest(PydanticBaseModel):
    maladie: str
    stade_vigne: str

class CalculRequest(PydanticBaseModel):
    operation: str
    parametres: Dict[str, Any] = Field(default_factory=dict)

class SessionInfo(PydanticBaseModel):
    thread_id: str
    messages_count: int
    created_at: Optional[str]
    last_activity: Optional[str]

class MessageInfo(PydanticBaseModel):
    index: int
    role: str
    content: str
    content_preview: str


# ─────────────────────────────────────────
# INITIALISATION
# ─────────────────────────────────────────
vigne_app_instance = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🌿 Démarrage VITI-AI | modèle={OLLAMA_MODEL} | ctx={OLLAMA_NUM_CTX} | timeout={OLLAMA_TIMEOUT}s")
    print(f"💾 Base SQLite : {DB_PATH}")
    vigne_app_instance["agent"] = build_vigne_agent(
        use_ollama=USE_OLLAMA,
        model_name=OLLAMA_MODEL,
        timeout=OLLAMA_TIMEOUT,
        num_ctx=OLLAMA_NUM_CTX,
        num_predict=OLLAMA_PREDICT,
        db_path=DB_PATH,
    )
    vigne_app_instance["db_path"] = DB_PATH
    print("✅ Agent prêt")
    yield
    print("👋 Arrêt VITI-AI")


# ─────────────────────────────────────────
# HELPERS SQLITE (lecture directe du DB)
# ─────────────────────────────────────────
def _get_db_path() -> str:
    return vigne_app_instance.get("db_path", DB_PATH)

def _list_threads_from_db() -> List[Dict]:
    """
    Lit directement la table 'checkpoints' de SqliteSaver pour lister
    tous les thread_id avec leur date de création et dernière activité.
    """
    db = _get_db_path()
    if not os.path.exists(db):
        return []
    try:
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        # SqliteSaver stocke dans la table 'checkpoints'
        cur.execute("""
            SELECT
                thread_id,
                COUNT(*) AS messages_count,
                MIN(checkpoint_id) AS created_at,
                MAX(checkpoint_id) AS last_activity
            FROM checkpoints
            GROUP BY thread_id
            ORDER BY last_activity DESC
        """)
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        return [{"erreur": str(e)}]

def _get_messages_from_state(agent, thread_id: str) -> List[Dict]:
    """Récupère les messages depuis le state LangGraph (MemorySaver ou SqliteSaver)."""
    config = {"configurable": {"thread_id": thread_id}}
    state = agent.get_state(config)
    if not state or not state.values:
        return []
    messages = state.values.get("messages", [])
    result = []
    for i, msg in enumerate(messages):
        role = type(msg).__name__.replace("Message", "").lower()
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        result.append({
            "index": i,
            "role": role,
            "content": content,
            "content_preview": content[:150] + ("..." if len(content) > 150 else ""),
        })
    return result


# ─────────────────────────────────────────
# GESTION ERREURS OLLAMA
# ─────────────────────────────────────────
def _ollama_error_message(err: str) -> str:
    if "wsarecv" in err or "forcibly closed" in err:
        return (
            "Ollama a fermé la connexion de force. Causes : "
            "(1) Modèle trop lourd pour la RAM/VRAM — essayez qwen2.5:7b ou llama3.2 ; "
            "(2) Génération trop longue — réduisez OLLAMA_PREDICT ; "
            "(3) Ollama planté — relancez avec 'ollama serve'."
        )
    if "connection refused" in err.lower() or "refused" in err.lower():
        return "Ollama n'est pas démarré. Lancez 'ollama serve' dans un terminal."
    if "timeout" in err.lower():
        return (
            f"Timeout après {OLLAMA_TIMEOUT}s. Augmentez OLLAMA_TIMEOUT ou "
            "utilisez un modèle plus rapide (ex: qwen2.5:3b, llama3.2)."
        )
    if "model" in err.lower() and ("not found" in err.lower() or "pull" in err.lower()):
        return f"Modèle '{OLLAMA_MODEL}' non disponible. Lancez : ollama pull {OLLAMA_MODEL}"
    return f"Erreur Ollama : {err[:300]}"


# ─────────────────────────────────────────
# APPLICATION
# ─────────────────────────────────────────
app = FastAPI(
    title="🍇 VITI-AI - Agent Agronome Viticole",
    description="""
## Système d'IA agentique multi-agent pour la viticulture

### Fonctionnalités
- **Diagnostic phytosanitaire** structuré via Output Parser Pydantic
- **Historique conversationnel** avec mémoire persistante (thread_id)
- **Outils intégrés** : météo, seuils d'alerte, calculs agronomiques
- **Pipeline LCEL** : prompt → model → parser

### Maladies gérées
Mildiou, Oïdium, Botrytis, Black-rot, Excoriose, Cochylis, Cicadelles, Acariens

### Architecture
`ChatPromptTemplate | ChatOllama(qwen3.5) | JsonOutputParser → DiagnosticVigne`
    """,
    version="1.1.0",
    lifespan=lifespan
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ─────────────────────────────────────────
# ENDPOINTS — SYSTÈME
# ─────────────────────────────────────────
@app.get("/health", tags=["Système"])
async def health_check():
    """État du service + configuration Ollama active."""
    return {
        "status": "✅ Opérationnel",
        "version": "1.2.0",
        "timestamp": datetime.now().isoformat(),
        "ollama_config": {
            "model": OLLAMA_MODEL,
            "url": OLLAMA_URL,
            "timeout_s": OLLAMA_TIMEOUT,
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": OLLAMA_PREDICT,
            "enabled": USE_OLLAMA,
        },
        "stockage": {
            "type": "SqliteSaver",
            "db_path": DB_PATH,
            "db_existe": os.path.exists(DB_PATH),
            "db_taille_ko": round(os.path.getsize(DB_PATH) / 1024, 1) if os.path.exists(DB_PATH) else 0,
        },
        "outils": ["get_meteo_vigne", "get_seuils_alerte", "calcul_agronomique"],
        "depannage": {
            "wsarecv_error": "Réduire OLLAMA_NUM_CTX=1024 ou changer de modèle (qwen2.5:7b)",
            "timeout_error": f"Augmenter OLLAMA_TIMEOUT (actuel: {OLLAMA_TIMEOUT}s)",
            "connection_refused": "Lancer 'ollama serve' dans un terminal",
        }
    }


@app.get("/ollama/check", tags=["Système"])
async def check_ollama():
    """Vérifie la connectivité Ollama et les modèles disponibles."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            models = r.json().get("models", [])
            names = [m["name"] for m in models]
            target_ok = any(OLLAMA_MODEL in n for n in names)
            return {
                "ollama_accessible": True,
                "url": OLLAMA_URL,
                "modeles_installes": names,
                "modele_cible": OLLAMA_MODEL,
                "modele_disponible": target_ok,
                "action_requise": None if target_ok else f"ollama pull {OLLAMA_MODEL}",
            }
    except Exception as e:
        return {
            "ollama_accessible": False,
            "url": OLLAMA_URL,
            "erreur": str(e),
            "action_requise": "ollama serve",
        }


# ─────────────────────────────────────────
# ENDPOINTS — AGENT
# ─────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
async def chat(request: ChatRequest):
    """
    Conversation avec l'agent agronome (mémoire via thread_id).

    L'historique est **persisté en SQLite** : une conversation reprend
    automatiquement là où elle s'était arrêtée, même après redémarrage.

    ### Exemples de messages
    - `"Mes feuilles ont des taches huileuses, que faire ?"`
    - `"Calculez la dose de bouillie bordelaise pour 3.5 ha"`
    - `"Quel est le risque météo mildiou cette semaine ?"`
    - `"Faites un diagnostic complet structuré de ma vigne"`
    """
    agent = vigne_app_instance.get("agent")
    if not agent:
        raise HTTPException(status_code=503, detail="Agent non initialisé")

    try:
        result = chat_avec_agent(
            app=agent,
            message=request.message,
            thread_id=request.thread_id,
            parcelle_context=request.parcelle_context,
        )
        return ChatResponse(
            response=result["response"] or "Aucune réponse générée.",
            diagnostic=result.get("diagnostic"),
            thread_id=result["thread_id"],
            messages_count=result["messages_count"],
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        err = str(e)
        raise HTTPException(
            status_code=500,
            detail={
                "erreur": _ollama_error_message(err),
                "technique": err[:400],
                "conseils": [
                    f"Vérifiez Ollama : curl {OLLAMA_URL}/api/tags",
                    f"Modèle actif : ollama list | grep {OLLAMA_MODEL}",
                    "Réduire le contexte : OLLAMA_NUM_CTX=1024",
                    "Changer de modèle : OLLAMA_MODEL=qwen2.5:7b",
                ]
            }
        )


# ─────────────────────────────────────────
# ENDPOINTS — DIAGNOSTIC
# ─────────────────────────────────────────
@app.get("/diagnostic/{thread_id}", tags=["Diagnostic"])
async def get_diagnostic(thread_id: str):
    """Récupère le dernier diagnostic structuré (DiagnosticVigne Pydantic)."""
    agent = vigne_app_instance.get("agent")
    if not agent:
        raise HTTPException(status_code=503, detail="Agent non initialisé")
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = agent.get_state(config)
        diag = state.values.get("diagnostic") if state else None
        return {
            "thread_id": thread_id,
            "diagnostic": diag,
            "message": None if diag else "Aucun diagnostic. Demandez un 'diagnostic complet'.",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────
# ENDPOINTS — DATA COMMONS
# ─────────────────────────────────────────
app.include_router(data_commons_router, prefix="/data-commons", tags=["data-commons"])

# ─────────────────────────────────────────
# ENDPOINTS — HISTORIQUE
# ─────────────────────────────────────────
@app.get("/historique", tags=["Historique"])
async def list_sessions():
    """
    Liste toutes les sessions de conversation stockées en SQLite.

    Retourne pour chaque session : thread_id, nombre de checkpoints,
    date de création et dernière activité (issues de la table LangGraph).
    """
    threads = _list_threads_from_db()
    return {
        "stockage": "SqliteSaver",
        "db_path": _get_db_path(),
        "sessions_count": len(threads),
        "sessions": threads,
    }


@app.get("/historique/{thread_id}", tags=["Historique"])
async def get_historique(thread_id: str):
    """
    Retourne l'historique complet des messages d'une session.

    Chaque message contient :
    - `role` : human | ai | tool | system
    - `content` : texte complet
    - `content_preview` : aperçu 150 caractères

    L'historique est lu depuis le **state LangGraph** reconstruit depuis SQLite.
    """
    agent = vigne_app_instance.get("agent")
    if not agent:
        raise HTTPException(status_code=503, detail="Agent non initialisé")

    try:
        messages = _get_messages_from_state(agent, thread_id)
        if not messages:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{thread_id}' introuvable ou vide."
            )

        config = {"configurable": {"thread_id": thread_id}}
        state = agent.get_state(config)
        parcelle = state.values.get("parcelle_context", {}) if state else {}
        has_diag = state.values.get("diagnostic") is not None if state else False

        return {
            "thread_id": thread_id,
            "messages_count": len(messages),
            "diagnostic_disponible": has_diag,
            "parcelle_context": parcelle,
            "historique": messages,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/historique/{thread_id}", tags=["Historique"])
async def delete_session(thread_id: str):
    """
    Supprime définitivement une session de la base SQLite.

    Supprime toutes les entrées `checkpoints` et `writes` associées
    au thread_id dans la base LangGraph.
    """
    db = _get_db_path()
    if not os.path.exists(db):
        raise HTTPException(status_code=404, detail="Base SQLite introuvable.")
    try:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        deleted_checkpoints = cur.rowcount
        # Supprimer aussi la table writes si elle existe
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='writes'"
        )
        if cur.fetchone():
            cur.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        conn.commit()
        conn.close()

        if deleted_checkpoints == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{thread_id}' introuvable dans la base SQLite."
            )
        return {
            "message": f"Session '{thread_id}' supprimée.",
            "checkpoints_supprimes": deleted_checkpoints,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────
# ENDPOINTS — OUTILS
# ─────────────────────────────────────────
@app.post("/tools/meteo", tags=["Outils"])
async def tool_meteo(request: MeteoRequest):
    """Météo actuelle + indices de risque phytosanitaires (Open-Meteo)."""
    try:
        return json.loads(get_meteo_vigne.invoke({
            "latitude": request.latitude,
            "longitude": request.longitude,
            "jours": request.jours,
        }))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/seuils", tags=["Outils"])
async def tool_seuils(request: SeuilRequest):
    """Seuils d'alerte et produits homologués pour une maladie/stade."""
    try:
        return json.loads(get_seuils_alerte.invoke({
            "maladie": request.maladie,
            "stade_vigne": request.stade_vigne,
        }))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/calcul", tags=["Outils"])
async def tool_calcul(request: CalculRequest):
    """
    Calculs agronomiques.

    ### Opérations disponibles
    - `degres_jours` → `temp_base`, `temp_moy_jour`, `jours`
    - `dose_traitement` → `surface_ha`, `dose_L_ha`, `concentration_pct`
    - `potentiel_alcool` → `densite`
    - `indice_mildiou` → `temperature`, `pluie_mm`, `humidite_pct`
    - `surface_feuillaire` → `longueur_feuille_cm`, `largeur_feuille_cm`
    """
    try:
        return json.loads(calcul_agronomique.invoke({
            "operation": request.operation,
            **request.parametres,
        }))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────
# ENDPOINTS — DOCUMENTATION
# ─────────────────────────────────────────
@app.get("/schema/diagnostic", tags=["Documentation"])
async def get_schema():
    """Schéma JSON complet du modèle DiagnosticVigne (Pydantic v2)."""
    return DiagnosticVigne.model_json_schema()