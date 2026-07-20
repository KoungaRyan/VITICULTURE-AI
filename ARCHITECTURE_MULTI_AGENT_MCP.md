# VITI-AI — Architecture cible : Multi-Agent via MCP

## 1. Constat sur l'existant

`agent/vigne_agent.py` construit **un seul** `StateGraph` avec **un seul** LLM (`ChatOllama`) sur lequel sont bindés les 6 tools de `tools/vigne_tools.py` :

| Tool | Domaine | Source de données |
|---|---|---|
| `get_meteo_vigne` | Météo | Open-Meteo (API externe) |
| `get_seuils_alerte` | Agronomie (statique) | Dict Python en dur |
| `calcul_agronomique` | Agronomie (calculs) | Formules locales |
| `get_contexte_parcelle` | Data Commons | SQLite (`data_commons/storage.py`) |
| `rechercher_connaissance_phytosanitaire` | RAG texte | Milvus (`connaissances_rag`) |
| `diagnostiquer_image_maladie` | RAG image | Milvus (`image_maladies`) |

C'est un pattern **ReAct classique** (un agent, N tools), pas un système multi-agent. D'où l'impression "un seul gros agent".

## 2. Architecture cible

```
                         ┌───────────────────────┐
                         │      SUPERVISOR        │
                         │ (StateGraph — routeur) │
                         └───────────┬────────────┘
              ┌───────────────┬──────┴───────┬────────────────┐
              ▼               ▼              ▼                ▼
      ┌───────────────┐┌──────────────┐┌──────────────┐┌───────────────┐
      │ Weather Agent  ││ Image Agent  ││ Text-RAG Agent││ (futur) Robot │
      │ react_agent    ││ react_agent  ││ react_agent   ││ Agent         │
      └───────┬────────┘└──────┬───────┘└──────┬───────┘└───────┬───────┘
              │ MCP client      │ MCP client     │ MCP client     │ MCP client
              ▼                 ▼                ▼                ▼
      ┌───────────────┐┌──────────────┐┌──────────────┐┌───────────────┐
      │ weather_server ││image_rag_    ││text_rag_     ││ robot_server  │
      │ .py (MCP)      ││server.py(MCP)││server.py(MCP)││ .py (MCP)     │
      │ get_meteo_vigne││diagnostiquer_││rechercher_   ││ capturer_image│
      │ get_seuils     ││image_maladie ││connaissance  ││ se_deplacer   │
      │ calcul_agro    ││ingerer_image ││ingerer_doc   ││ declencher_tr.│
      └───────────────┘└──────────────┘└──────────────┘└───────────────┘
```

**Principe clé** : chaque domaine métier devient un **serveur MCP indépendant** (un process qui expose ses tools via le protocole MCP), et chaque agent spécialisé est un **client MCP** qui ne voit *que* les tools de son domaine. Le Supervisor ne connaît pas les tools eux-mêmes — il ne voit que 3 (puis 4) agents, qu'il appelle comme des sous-outils.

**Pourquoi c'est le bon choix pour le robot** : le jour où le robot arrive, il n'a besoin de rien connaître de LangGraph, d'Ollama ni de Milvus. Il expose juste `robot_server.py` avec ses propres tools MCP (`capturer_image`, `se_deplacer_vers`, `declencher_traitement`...). Comme tout le système parle déjà MCP, l'intégration du robot est une **addition**, pas une refonte : un serveur MCP de plus, un agent de plus, une route de plus dans le Supervisor.

## 3. Répartition des tools existants

- **Weather Agent** ← `get_meteo_vigne`, `get_seuils_alerte`, `calcul_agronomique`
  (regroupés : ce sont les 3 tools qui alimentent l'évaluation du risque phytosanitaire à partir de données météo/agro, aucun n'a besoin de Milvus)
- **Image RAG Agent** ← `diagnostiquer_image_maladie`, `ingerer_image_reference`
- **Text RAG Agent** ← `rechercher_connaissance_phytosanitaire`, `ingerer_document`
- **Data Commons** ← `get_contexte_parcelle` : à débattre — soit un 4ᵉ agent séparé (propre, mais un agent de plus à router), soit un tool que le Supervisor garde pour lui-même côté "contexte" avant de router (plus simple, car il sert à *tous* les agents). Recommandation : le garder côté Supervisor pour l'instant, migrable en agent séparé plus tard si besoin.

## 4. Nouvelle arborescence

```
VITICULTURE-AI/
├── mcp_servers/
│   ├── weather_server.py        # FastMCP — expose get_meteo_vigne, get_seuils_alerte, calcul_agronomique
│   ├── image_rag_server.py      # FastMCP — expose diagnostiquer_image_maladie, ingerer_image_reference
│   ├── text_rag_server.py       # FastMCP — expose rechercher_connaissance_phytosanitaire, ingerer_document
│   └── robot_server.py          # (futur) FastMCP côté robot
├── agents/
│   ├── weather_agent.py         # create_react_agent + MultiServerMCPClient → weather_server
│   ├── image_agent.py           # create_react_agent + MultiServerMCPClient → image_rag_server
│   ├── text_rag_agent.py        # create_react_agent + MultiServerMCPClient → text_rag_server
│   └── supervisor.py            # StateGraph qui route vers les agents ci-dessus
├── tools/vigne_tools.py         # conservé : la logique métier brute, réutilisée par les serveurs MCP
├── embeddings/ , vector_store/ , data_commons/   # inchangés
├── api/main.py                  # adapté : appelle supervisor au lieu de build_vigne_agent
```

## 5. Dépendances à ajouter

```txt
mcp==1.*                      # SDK officiel MCP (serveurs FastMCP + client)
langchain-mcp-adapters==0.1.* # pont MCP → tools LangChain (MultiServerMCPClient)
langgraph-supervisor==0.0.*   # optionnel : create_supervisor() prêt à l'emploi
```

## 6. Migration étape par étape

1. **Ajouter les dépendances** ci-dessus à `requirements.txt`.
2. **Créer `mcp_servers/`** : pour chaque domaine, un script `FastMCP` qui ré-expose les fonctions déjà écrites dans `tools/vigne_tools.py` (on retire juste le décorateur `@tool` de LangChain et on les enregistre avec `@mcp.tool()` — la logique métier ne change pas).
3. **Tester chaque serveur isolément** avec `mcp dev mcp_servers/weather_server.py` (inspecteur MCP officiel) avant de brancher quoi que ce soit.
4. **Créer les 3 agents** (`agents/weather_agent.py`, etc.) : chacun un `create_react_agent` (LangGraph prebuilt) avec un prompt système *court et spécialisé*, connecté uniquement à son serveur via `MultiServerMCPClient`.
5. **Créer `agents/supervisor.py`** : le `StateGraph` qui décide quel agent appeler (le pattern "Supervisor" que tu as choisi — chaque agent est vu comme un tool par le LLM superviseur).
6. **Adapter `api/main.py`** : remplacer `build_vigne_agent()`/`chat_avec_agent()` par le nouveau supervisor, en gardant la même signature de fonction côté FastAPI pour ne rien casser côté clients de l'API.
7. **(Plus tard) `robot_server.py`** : quand le robot est prêt, même recette — un serveur MCP + un agent + une route supervisor.

---

Je te propose de commencer par l'étape 2 (les 3 serveurs MCP) directement sur ton repo cloné localement — c'est le morceau le plus mécanique et ça ne casse rien d'existant tant que le supervisor n'est pas branché à l'API.
