"""
API REST FastAPI — Agent Agronome Viticole
==========================================
Étape 4 : Exposition de l'agent via une API REST.
Endpoints : /chat, /diagnostic, /tools/meteo, /tools/calcul, /health
"""
import json
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as PydanticBaseModel, Field

sys.path.insert(0, '/home/claude/vigne_agent')
from agent.vigne_agent import build_vigne_agent, chat_avec_agent
from tools.vigne_tools import get_meteo_vigne, get_seuils_alerte, calcul_agronomique
from models.schemas import DiagnosticVigne


# ─────────────────────────────────────────
# MODÈLES DE REQUÊTE/RÉPONSE (FastAPI)
# ─────────────────────────────────────────
class ChatRequest(PydanticBaseModel):
    message: str = Field(..., description="Message de l'utilisateur")
    thread_id: str = Field(default="session_001", description="ID de session pour la mémoire")
    parcelle_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Contexte parcelle (cepage, localisation, stade...)"
    )

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
# INITIALISATION DE L'APP
# ─────────────────────────────────────────
vigne_app_instance = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialisation et nettoyage de l'application."""
    print("🌿 Démarrage de VITI-AI Agent...")
    vigne_app_instance["agent"] = build_vigne_agent(
        use_ollama=False,  # Mettre True si Ollama est installé
        model_name="llama3.2"
    )
    print("✅ Agent viticole prêt !")
    yield
    print("👋 Arrêt de VITI-AI Agent")


app = FastAPI(
    title="🍇 VITI-AI — Agent Agronome Viticole",
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────
@app.get("/health", tags=["Système"])
async def health_check():
    """Vérification de l'état du service."""
    return {
        "status": "✅ Opérationnel",
        "agent": "VITI-AI Agent Viticole",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "outils_disponibles": [
            "get_meteo_vigne",
            "get_seuils_alerte", 
            "calcul_agronomique"
        ],
        "maladies_repertoriees": [
            "Mildiou (Plasmopara viticola)",
            "Oïdium (Erysiphe necator)",
            "Botrytis (Botrytis cinerea)",
            "Black-rot (Guignardia bidwellii)",
            "Excoriose (Phomopsis viticola)",
            "Cochylis (Eupoecilia ambiguella)",
            "Acariens"
        ]
    }


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
async def chat(request: ChatRequest):
    """
    Conversation avec l'agent agronome viticole.
    
    L'historique est conservé grâce au **thread_id** (mémoire LangGraph).
    Chaque session identifiée par un thread_id différent dispose de son propre historique.
    
    ### Exemples de messages
    - "Mes feuilles présentent des taches huileuses, que faire ?"
    - "Quels sont les risques météo cette semaine pour le mildiou ?"
    - "Calculez la dose de bouillie bordelaise pour 3.5 ha"
    - "Faites un diagnostic complet de ma vigne"
    """
    agent = vigne_app_instance.get("agent")
    if not agent:
        raise HTTPException(status_code=503, detail="Agent non initialisé")
    
    try:
        result = chat_avec_agent(
            app=agent,
            message=request.message,
            thread_id=request.thread_id,
            parcelle_context=request.parcelle_context
        )
        
        return ChatResponse(
            response=result["response"] or "Je n'ai pas pu générer de réponse.",
            diagnostic=result.get("diagnostic"),
            thread_id=result["thread_id"],
            messages_count=result["messages_count"],
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur agent : {str(e)}")


@app.get("/diagnostic/{thread_id}", tags=["Diagnostic"])
async def get_diagnostic(thread_id: str):
    """
    Récupère le dernier diagnostic structuré pour une session.
    
    Retourne un objet **DiagnosticVigne** conforme au schéma Pydantic.
    """
    agent = vigne_app_instance.get("agent")
    if not agent:
        raise HTTPException(status_code=503, detail="Agent non initialisé")
    
    # Récupération de l'état depuis le checkpointer
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = agent.get_state(config)
        if state and state.values.get("diagnostic"):
            return {
                "thread_id": thread_id,
                "diagnostic": state.values["diagnostic"],
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "thread_id": thread_id,
                "diagnostic": None,
                "message": "Aucun diagnostic structuré disponible. Demandez un 'diagnostic complet'."
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/meteo", tags=["Outils"])
async def tool_meteo(request: MeteoRequest):
    """Outil météo : conditions actuelles et risques phytosanitaires."""
    try:
        result = get_meteo_vigne.invoke({
            "latitude": request.latitude,
            "longitude": request.longitude,
            "jours": request.jours
        })
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/seuils", tags=["Outils"])
async def tool_seuils(request: SeuilRequest):
    """Outil seuils : informations et seuils d'alerte pour une maladie/stade."""
    try:
        result = get_seuils_alerte.invoke({
            "maladie": request.maladie,
            "stade_vigne": request.stade_vigne
        })
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/calcul", tags=["Outils"])
async def tool_calcul(request: CalculRequest):
    """
    Outil de calcul agronomique.
    
    ### Opérations disponibles
    - `degres_jours` : DJC (temp_base, temp_moy_jour, jours)
    - `dose_traitement` : Dose produit (surface_ha, dose_L_ha, concentration_pct)  
    - `potentiel_alcool` : Degré alcoolique (densite)
    - `indice_mildiou` : Risque EPI (temperature, pluie_mm, humidite_pct)
    - `surface_feuillaire` : LAI (longueur_feuille_cm, largeur_feuille_cm, nb_feuilles_par_rameau)
    """
    try:
        result = calcul_agronomique.invoke({
            "operation": request.operation,
            **request.parametres
        })
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/schema/diagnostic", tags=["Documentation"])
async def get_diagnostic_schema():
    """Retourne le schéma JSON complet du modèle DiagnosticVigne."""
    return DiagnosticVigne.model_json_schema()
