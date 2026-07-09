# Data Commons — VITI-AI

Étape 1 de la mise en place du DATA COMMONS viticole (données internes privées
Mialtech + données publiques + données capteurs/robot), selon la fiche de stage.

## Installation dans votre projet

1. Copiez le dossier `data_commons/` à la racine de `VITI-AI`, à côté de
   `models/`, `tools/`, `agent/`, `api/`.
2. Ajoutez `requests` à vos dépendances (déjà probablement présent via vos tools).
3. Variable d'environnement optionnelle (suit le même pattern que `DB_PATH`
   pour le checkpointer) :
   ```
   DATA_COMMONS_DB_PATH=data/data_commons.sqlite
   ```

## Brancher sur votre API existante (`api/main.py`)

```python
from data_commons.api_router import router as data_commons_router
app.include_router(data_commons_router, prefix="/data-commons", tags=["data-commons"])
```

Nouveaux endpoints :
- `POST /data-commons/parcelles` — créer/mettre à jour une parcelle
- `GET  /data-commons/parcelles/{id}/contexte` — historique agrégé (météo,
  maturation, maladies, traitements) → **à consommer par votre agent
  LangGraph** comme contexte avant diagnostic
- `POST /data-commons/ingestion/run` — déclenche une passe d'ingestion

## Brancher sur l'agent LangGraph (`agent/vigne_agent.py`)

Le plus simple : ajoutez un 4ᵉ tool `get_contexte_parcelle` qui appelle
`DataCommonsStore.get_historique_parcelle(id_parcelle)` et l'exposez à côté de
vos 3 tools existants (`get_meteo_vigne`, `get_seuils_alerte`,
`calcul_agronomique`). L'agent peut alors croiser météo temps réel + historique
capteurs + historique maladies avant de produire un `DiagnosticVigne`.

```python
from langchain_core.tools import tool
from data_commons.storage import DataCommonsStore

_store = DataCommonsStore()

@tool
def get_contexte_parcelle(id_parcelle: str) -> dict:
    """Retourne l'historique météo/maturation/maladies/traitements d'une parcelle."""
    return _store.get_historique_parcelle(id_parcelle)
```

## Étapes suivantes (roadmap Data Commons)

1. ✅ Schémas unifiés (`schemas.py`)
2. ✅ Stockage SQLite (`storage.py`) — migration Postgres/Milvus prévue sans
   changer l'interface (mêmes méthodes `add_*`, `get_*`)
3. ✅ Connecteurs : météo publique (Open-Meteo), capteurs IoT (CSV → MQTT plus
   tard), Mialtech (CSV → API si disponible)
4. ✅ Pipeline d'ingestion orchestré + endpoint API
5. ⏭️ **Prochaine étape recommandée** : brancher un vrai flux capteurs
   (Raspberry Pi 5 / Arduino Mega / RS-485) à la place du CSV stub
6. ⏭️ Vector store (Milvus) pour le RAG et l'Image Agent (nomic-embed-vision)
   — le Data Commons SQL sert de source de vérité structurée, le vector store
   viendra en complément pour la similarité d'images et le RAG documentaire
7. ⏭️ Étude comparative Oracle 26AI / Postgres / Milvus (mission dédiée de la
   fiche de stage)
