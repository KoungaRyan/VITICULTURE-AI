# Contenu de ce zip

Ce zip contient uniquement ce qui manque dans ton repo GitHub actuel pour
activer l'architecture multi-agent MCP (Weather / Image-RAG / Text-RAG +
Supervisor).

## Comment l'intégrer dans ton repo local

1. Décompresse ce zip.
2. Copie les dossiers `agents/` et `mcp_servers/` **à la racine de ton repo**
   VITICULTURE-AI (ce sont des dossiers entièrement nouveaux, rien à fusionner).
3. Remplace ton `api/main.py` par celui fourni ici — OU applique le patch
   `CHANGEMENTS_api_main_et_requirements.diff` avec :
   ```
   git apply CHANGEMENTS_api_main_et_requirements.diff
   ```
   (place le fichier .diff à la racine du repo avant de lancer la commande)
4. Remplace ton `requirements.txt` par celui fourni ici (ou applique le même
   patch, il couvre aussi ce fichier), puis :
   ```
   pip install -r requirements.txt
   ```
5. Vérifie que chaque serveur MCP démarre seul avant de tout lancer ensemble :
   ```
   python mcp_servers/weather_server.py
   ```
   (Ctrl+C pour arrêter — juste un test de démarrage, pas d'inspecteur ici)
6. Relance l'API normalement (`uvicorn api.main:app ...`) : au démarrage tu
   dois voir les 3 sous-agents s'initialiser (chacun spawn son propre serveur
   MCP en sous-processus stdio) avant que l'API accepte les requêtes.
7. Une fois que tu as vérifié que tout fonctionne : `git add agents/
   mcp_servers/ api/main.py requirements.txt && git commit -m "Migration
   architecture multi-agent MCP (Weather/Image/Text-RAG + Supervisor)" &&
   git push`.

## Ce qui n'a PAS changé
`tools/vigne_tools.py`, `rag/`, `image_agent/`, `vector_store/`,
`data_commons/`, `embeddings/`, `models/schemas.py` — tout ça reste utilisé
tel quel, réutilisé par les nouveaux serveurs MCP sans duplication de code.

## Prochaine étape (robot)
Quand le robot sera prêt : même recette — `mcp_servers/robot_server.py`
(FastMCP, tools `capturer_image`/`se_deplacer_vers`/`declencher_traitement`),
`agents/robot_agent.py` (même pattern que les 3 autres), puis ajouter
`consulter_agent_robot` dans `agents/supervisor.py` (SUPERVISOR_TOOLS +
prompt). Rien d'autre à changer côté architecture.
