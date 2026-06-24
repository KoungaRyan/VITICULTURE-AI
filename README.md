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

### Configuration dans `agent/vigne_agent.py`

```python
# Basculer entre simulation et Ollama
agent = build_vigne_agent(
    use_ollama=True,        # False = mode simulation
    model_name="llama3.2"  # ou "mistral", "llama3.1", etc.
)
```

---

## Exemple d'Utilisation

### Conversation avec mémoire

```python
from agent.vigne_agent import build_vigne_agent, chat_avec_agent

agent = build_vigne_agent(use_ollama=True)

# Tour 1 — L'historique commence
r1 = chat_avec_agent(
    agent,
    "Mes feuilles ont des taches huileuses en floraison",
    thread_id="exploitation_dupont",
    parcelle_context={"localisation": "Bordeaux", "cepage": "Merlot"}
)

# Tour 2 — L'agent se souvient du contexte
r2 = chat_avec_agent(
    agent,
    "Il a plu 12mm ce matin, quel est le risque ?",
    thread_id="exploitation_dupont"  # Même thread = mémoire conservée
)

# Tour 3 — Diagnostic structuré Pydantic
r3 = chat_avec_agent(
    agent,
    "Faites un diagnostic complet structuré",
    thread_id="exploitation_dupont"
)
print(r3["diagnostic"])  # → DiagnosticVigne validé
```

### API REST

```bash
# Chat avec mémoire
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Taches huileuses sur mes vignes", "thread_id": "session_001"}'

# Météo + risques phytosanitaires
curl -X POST http://localhost:8000/tools/meteo \
  -d '{"latitude": 44.83, "longitude": -0.57, "jours": 3}'

# Calcul dose traitement
curl -X POST http://localhost:8000/tools/calcul \
  -d '{"operation": "dose_traitement", "parametres": {"surface_ha": 3.5, "dose_L_ha": 2.5}}'
```

---

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

## Calculs Agronomiques Intégrés

```python
# Degrés-Jours de Croissance
calcul_agronomique("degres_jours", temp_base=10, temp_moy_jour=22, jours=1)
# → 12.0 °C.jours

# Dose de traitement
calcul_agronomique("dose_traitement", surface_ha=3.5, dose_L_ha=2.5)
# → 8.75 L de produit pour 3.5 ha

# Potentiel alcoolique
calcul_agronomique("potentiel_alcool", densite=1082)
# → 12.3% vol.

# Indice de risque Mildiou (EPI)
calcul_agronomique("indice_mildiou", temperature=20, pluie_mm=8, humidite_pct=85)
# → EPI: 8.3 (Faible)

# Surface Foliaire Exposée (LAI)
calcul_agronomique("surface_feuillaire", longueur_feuille_cm=15, largeur_feuille_cm=14)
# → LAI: 1.85 m²/m²
```

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
