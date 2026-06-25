"""
API REST FastAPI — Agent Agronome Viticole VITI-AI
===================================================
Corrections robustesse v1.1 :
  - Timeout configurable via variables d'environnement
  - Gestion fine des erreurs Ollama (wsarecv, timeout, connexion refusée)
  - Messages d'erreur clairs avec diagnostic de la cause
  - Paramètres Ollama exposés dans /health
"""
import json
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as PydanticBaseModel, Field

sys.path.insert(0, '/home/claude/vigne_agent')
from agent.vigne_agent import build_vigne_agent, chat_avec_agent
from tools.vigne_tools import get_meteo_vigne, get_seuils_alerte, calcul_agronomique
from models.schemas import DiagnosticVigne


# ─────────────────────────────────────────
# CONFIG (modifiable via variables d'env)
# ─────────────────────────────────────────
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "llama3.2")   # llama3.2 8B par défaut
OLLAMA_URL      = os.getenv("OLLAMA_URL",      "http://localhost:11434")
OLLAMA_TIMEOUT  = int(os.getenv("OLLAMA_TIMEOUT",  "180"))   # 180s — llama3.2 peut être lent
OLLAMA_NUM_CTX  = int(os.getenv("OLLAMA_NUM_CTX",  "2048"))  # 2024 → stable avec 8B sur GPU limité
OLLAMA_PREDICT  = int(os.getenv("OLLAMA_PREDICT",  "512"))   # 512 → évite les crashes mid-stream
USE_OLLAMA      = os.getenv("USE_OLLAMA", "true").lower() == "true"


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


# ─────────────────────────────────────────
# INITIALISATION
# ─────────────────────────────────────────
vigne_app_instance = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🌿 Démarrage VITI-AI | modèle={OLLAMA_MODEL} | ctx={OLLAMA_NUM_CTX} | timeout={OLLAMA_TIMEOUT}s")
    vigne_app_instance["agent"] = build_vigne_agent(
        use_ollama=USE_OLLAMA,
        model_name=OLLAMA_MODEL,
        timeout=OLLAMA_TIMEOUT,
        num_ctx=OLLAMA_NUM_CTX,
        num_predict=OLLAMA_PREDICT,
    )
    print("✅ Agent prêt")
    yield
    print("👋 Arrêt VITI-AI")


def _ollama_error_message(err: str) -> str:
    """Traduit les erreurs Ollama en messages lisibles."""
    if "wsarecv" in err or "forcibly closed" in err:
        return (
            "Ollama a fermé la connexion de force. Causes probables : "
            "(1) Modèle trop lourd pour la RAM/VRAM disponible — essayez un modèle plus petit (ex: llama3.2:1b) ; "
            "(2) Génération trop longue — réduisez OLLAMA_PREDICT ; "
            "(3) Ollama planté — relancez avec 'ollama serve'."
        )
    if "connection refused" in err.lower() or "refused" in err.lower():
        return "Ollama n'est pas démarré. Lancez 'ollama serve' dans un terminal."
    if "timeout" in err.lower():
        return (
            f"Timeout après {OLLAMA_TIMEOUT}s. Augmentez OLLAMA_TIMEOUT ou "
            "utilisez un modèle plus rapide (ex: llama3.2:1b, phi3:mini)."
        )
    if "model" in err.lower() and ("not found" in err.lower() or "pull" in err.lower()):
        return f"Modèle '{OLLAMA_MODEL}' non disponible. Lancez : ollama pull {OLLAMA_MODEL}"
    return f"Erreur Ollama : {err[:300]}"


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
`ChatPromptTemplate | ChatOllama(llama3.2) | JsonOutputParser → DiagnosticVigne`
    """,
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ─────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────
@app.get("/health", tags=["Système"])
async def health_check():
    """État du service + configuration Ollama active."""
    return {
        "status": "✅ Opérationnel",
        "version": "1.1.0",
        "timestamp": datetime.now().isoformat(),
        "ollama_config": {
            "model": OLLAMA_MODEL,
            "url": OLLAMA_URL,
            "timeout_s": OLLAMA_TIMEOUT,
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": OLLAMA_PREDICT,
            "enabled": USE_OLLAMA,
        },
        "outils": ["get_meteo_vigne", "get_seuils_alerte", "calcul_agronomique"],
        "depannage": {
            "wsarecv_error": "llama3.2 8B crash → réduire OLLAMA_NUM_CTX=512 ou OLLAMA_PREDICT=150",
            "timeout_error": f"Augmenter OLLAMA_TIMEOUT (actuel: {OLLAMA_TIMEOUT}s)",
            "connection_refused": "Lancer 'ollama serve' dans un terminal",
        }
    }


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
async def chat(request: ChatRequest):
    """
    Conversation avec l'agent agronome (mémoire via thread_id).

    ### Exemples de messages
    - "Mes feuilles ont des taches huileuses, que faire ?"
    - "Calculez la dose de bouillie bordelaise pour 3.5 ha"`
    - "Quel est le risque météo mildiou cette semaine ?"
    - "Faites un diagnostic complet structuré de ma vigne"
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
        friendly = _ollama_error_message(err)
        raise HTTPException(
            status_code=500,
            detail={
                "erreur": friendly,
                "technique": err[:400],
                "conseils": [
                    f"Vérifiez qu'Ollama tourne : curl {OLLAMA_URL}/api/tags",
                    f"Vérifiez le modèle : ollama list | grep {OLLAMA_MODEL}",
                    "Réduisez le contexte : OLLAMA_NUM_CTX=1024 uvicorn ...",
                    "Utilisez un modèle plus léger : OLLAMA_MODEL=llama3.2:1b uvicorn ...",
                ]
            }
        )


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


@app.post("/tools/meteo", tags=["Outils"])
async def tool_meteo(request: MeteoRequest):
    """Météo actuelle + indices de risque phytosanitaires."""
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

    ### Opérations
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


@app.get("/schema/diagnostic", tags=["Documentation"])
async def get_schema():
    """Schéma JSON complet du modèle DiagnosticVigne (Pydantic)."""
    return DiagnosticVigne.model_json_schema()


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