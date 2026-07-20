# 🍇 VITI-AI — Agent Agronome Viticole Multi-Agent

> Système d'IA agentique pour la détection des maladies de la vigne et l'optimisation des décisions phytosanitaires

---

## Architecture du Système

```
ChatPromptTemplate ──► ChatOllama (llama3.2) ──► JsonOutputParser
       │                        │                        │
  Système agrono.          Tools binding           DiagnosticVigne
  Variables context        (ReAct pattern)          (Pydantic v2)
```

### Graphe LangGraph

```
            [START]
               │
               ▼
         ┌─────────────┐
         │    AGENT    │  ← ChatOllama + MemorySaver (thread_id)
         │  (llama3.2) │
         └──────┬───────┘
                │
   ┌────────────┼──────────────┐
   ▼            ▼              ▼
[TOOLS]  [EXTRACT_DIAG]      [END]
meteo    LCEL chain
seuils   → Pydantic
calcul   → DiagnosticVigne
   │            │
   └──► AGENT   END
```

---

## Structure du Projet

```
vigne_agent/
├── models/
│   └── schemas.py          # Pydantic : DiagnosticVigne, Maladie, TraitementRecommande
├── tools/
│   └── vigne_tools.py      # 3 outils : météo, seuils, calculs agronomiques
├── agent/
│   └── vigne_agent.py      # LangGraph + LCEL + mémoire + prompts
├── api/
│   └── main.py             # API REST FastAPI
├── demo.py                 # Démonstration complète (5 étapes)
└── README.md               # Ce fichier
```

---

## Étapes de Mise en Œuvre

### Étape 1 — Schémas Pydantic (`models/schemas.py`)

Définition des modèles de données structurées pour le diagnostic :

```python
class DiagnosticVigne(BaseModel):
    stade_phenologique: StadeVigne
    maladies_detectees: List[Maladie]
    severite_globale: SeveriteEnum      # faible|moderée|élevée|critique
    risque_propagation: float           # 0.0 → 1.0
    traitements_recommandes: List[TraitementRecommande]
    alerte: bool                        # Intervention urgente
    confidence_score: float             # 0.0 → 1.0
```

### Étape 2 — Outils (`tools/vigne_tools.py`)

3 outils décorés `@tool` pour le ReAct agent :

| Outil | Description | Paramètres |
|-------|-------------|------------|
| `get_meteo_vigne` | Météo + risques phytosanitaires (Open-Meteo) | lat, lon, jours |
| `get_seuils_alerte` | Base de données phytosanitaire officielle | maladie, stade |
| `calcul_agronomique` | 5 calculs : DJC, dose, alcool, EPI mildiou, LAI | operation, **kwargs |

### Étape 3 — Pipeline LCEL (`agent/vigne_agent.py`)

Trois pipelines LCEL en cascade :

```python
# Pipeline 1 : Réponse texte
chain_str = prompt | llm | StrOutputParser()

# Pipeline 2 : Extraction JSON
chain_json = prompt | llm | JsonOutputParser()

# Pipeline 3 : Enrichissement
chain_enrichi = (
    RunnablePassthrough.assign(question=lambda x: x["question"])
    | prompt | llm | JsonOutputParser()
    | RunnableLambda(lambda d: {**d, "source": "VITI-AI", "confiance": 0.85})
)
```

### Étape 4 — Agent + Mémoire (`agent/vigne_agent.py`)

```python
# Graphe LangGraph avec MemorySaver
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# Conversation avec mémoire persistante
config = {"configurable": {"thread_id": "vigneron_dupont_A"}}
result = app.invoke(state, config=config)
```

### Étape 5 — API REST (`api/main.py`)

```
POST /chat              → Conversation + mémoire (thread_id)
GET  /diagnostic/{id}  → Diagnostic structuré Pydantic
POST /tools/meteo      → Conditions météo + risques
POST /tools/seuils     → Seuils d'alerte phytosanitaires
POST /tools/calcul     → Calculs agronomiques
GET  /schema/diagnostic → Schéma JSON DiagnosticVigne
GET  /health           → État du service
```

---

## Installation & Démarrage

### Prérequis

```bash
pip install langchain langchain-community langchain-core langgraph \
            langchain-ollama pydantic fastapi uvicorn requests
```

### Avec Ollama (Production)

```bash
# Installation Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Téléchargement du modèle
ollama pull llama3.2

# Lancement de l'agent
cd vigne_agent
python demo.py
```

### Démarrage de l'API

```bash
cd vigne_agent
uvicorn api.main:app --reload --port 8000

# Documentation interactive
open http://localhost:8000/docs
```





## Maladies et Ravageurs Gérés

| Maladie | Agent pathogène | Stades sensibles |
|---------|----------------|-----------------|
| **Mildiou** | *Plasmopara viticola* | Feuillaison, floraison, nouaison |
| **Oïdium** | *Erysiphe necator* | Débourrement → nouaison |
| **Botrytis** | *Botrytis cinerea* | Floraison, véraison, maturité |
| **Black-rot** | *Guignardia bidwellii* | Feuillaison → nouaison |
| **Excoriose** | *Phomopsis viticola* | Débourrement |
| **Cochylis** | *Eupoecilia ambiguella* | G1=floraison, G2=fermeture grappe |
| **Acariens** | *Panonychus ulmi* | Débourrement, feuillaison |

---



## Technologies Utilisées

| Composant | Technologie |
|-----------|------------|
| LLM local | Ollama + llama3.2 |
| Orchestration | LangGraph (StateGraph) |
| Pipeline | LCEL (LangChain Expression Language) |
| Mémoire | MemorySaver (checkpointer in-memory) |
| Outils | `@tool` decorator + ToolNode |
| Schémas | Pydantic v2 |
| API | FastAPI + Uvicorn |
| Météo | Open-Meteo (API gratuite) |

---

*VITI-AI v1.0 — Système d'IA agentique pour la viticulture*
