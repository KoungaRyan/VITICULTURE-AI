"""
Agent Agronome Viticole — Architecture LangGraph + LCEL
========================================================
Corrections robustesse :
  - Timeout Ollama configurable (120s)
  - Troncature du contexte pour éviter les dépassements de fenêtre
  - Retry automatique sur erreur de connexion
  - Fallback gracieux si Ollama coupe la connexion
  - Historique limité pour éviter les contextes trop longs
"""
import json
import os
from typing import Annotated, Sequence, TypedDict, Optional
from datetime import datetime

from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import  RunnableLambda

from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
#from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode

import sys
sys.path.insert(0, '/home/claude/vigne_agent')
from tools.vigne_tools import VIGNE_TOOLS
from models.schemas import DiagnosticVigne


# ─────────────────────────────────────────────────────────────
# ÉTAT DU GRAPHE
# ─────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    thread_id: str
    mode: str
    diagnostic: Optional[dict]
    parcelle_context: Optional[dict]


# ─────────────────────────────────────────────────────────────
# PROMPTS (volontairement courts pour limiter les tokens)
# ─────────────────────────────────────────────────────────────
SYSTEM_AGRONOME = """Tu es VITI-AI, expert viticole. Réponds en français, sois concis (max 150 mots).
Date: {date_actuelle}
Parcelle: {parcelle_context}
Maladies gérées: mildiou, oïdium, botrytis, black-rot, excoriose, cochylis, acariens.
Utilise tes outils pour météo/seuils/calculs. Rappelle les DAR. Propose alternatives bio."""

PROMPT_AGENT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_AGRONOME),
    MessagesPlaceholder(variable_name="messages"),
])

PROMPT_DIAGNOSTIC_SYSTEM = (
    "Extrais un diagnostic JSON depuis la conversation. "
    "Retourne UNIQUEMENT du JSON brut (sans markdown, sans backtick). "
    "Champs requis : parcelle_id, cepage, stade_phenologique (feuillaison/floraison/veraison/maturite/dormance), "
    "date_diagnostic, maladies_detectees (liste avec nom/probabilite/symptomes_observes/organes_touches), "
    "severite_globale (faible/moderee/elevee/critique), risque_propagation (0-1), "
    "traitements_recommandes (liste avec produit/type_traitement/dose/timing/precautions/delai_avant_recolte), "
    "actions_culturales, surveillance, resume_diagnostic, alerte (bool), confidence_score (0-1)."
)

PROMPT_DIAGNOSTIC = ChatPromptTemplate.from_messages([
    ("system", PROMPT_DIAGNOSTIC_SYSTEM),
    ("human", "Conversation :\n{conversation}\n\nMeteo :\n{meteo_context}"),
])



# ─────────────────────────────────────────────────────────────
# INITIALISATION DU MODÈLE OLLAMA (avec timeout et retry)
# ─────────────────────────────────────────────────────────────
def get_model(
    use_ollama: bool = True,
    model_name: str = "qwen3.5:4b",
    timeout: int = 180,
    num_ctx: int = 1024,       # Réduit au max → moins de RAM → stable avec llama3.1 8B
    num_predict: int = 300,    # Court → évite les crashes mid-stream
) -> object:
    """
    Initialise ChatOllama avec des paramètres robustes.
    
    Paramètres clés contre les déconnexions :
    - timeout : délai max par requête (120s)
    - num_ctx : taille contexte (2048 évite les OOM sur GPU limité)
    - num_predict : limite la génération (512 tokens max par réponse)
    - keep_alive : garde le modèle chargé entre les requêtes ("5m")
    """
    if use_ollama:
        try:
            model = ChatOllama(
                model=model_name,
                temperature=0.1,
                base_url="http://localhost:11434",
                timeout=timeout,
                num_ctx=num_ctx,
                num_predict=num_predict,
                keep_alive="5m",       # Évite le rechargement du modèle entre requêtes
                request_timeout=timeout,
            )
            # Test rapide de connexion (message très court)
            test = model.invoke([HumanMessage(content="ok")])
            print(f"✅ Ollama ({model_name}) connecté | ctx={num_ctx} | predict={num_predict}")
            return model
        except Exception as e:
            err = str(e)
            if "wsarecv" in err or "connection" in err.lower() or "refused" in err.lower():
                print(f"⚠️  Ollama déconnecté ({err[:80]})")
                print("   → Vérifiez que Ollama tourne : 'ollama serve'")
                print("   → Vérifiez le modèle : 'ollama list'")
            else:
                print(f"⚠️  Ollama non disponible : {err[:100]}")
            print("   → Passage en mode simulation")

    return MockViticulureModel()


def invoke_with_retry(model, messages, max_retries: int = 2) -> BaseMessage:
    """
    Invoque le modèle avec retry automatique sur erreur réseau.
    Tronque le contexte si trop long.
    """
    # Limiter l'historique à 4 messages max pour éviter les dépassements de contexte
    if len(messages) > 5:
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
        messages = system_msgs + other_msgs[-4:]

    for attempt in range(max_retries + 1):
        try:
            return model.invoke(messages)
        except Exception as e:
            err = str(e)
            is_connection_err = any(k in err for k in [
                "wsarecv", "wsasend", "forcibly closed", "connection",
                "timeout", "EOF", "broken pipe", "reset"
            ])
            if is_connection_err and attempt < max_retries:
                import time
                wait = (attempt + 1) * 2
                print(f"⚠️  Erreur réseau Ollama (tentative {attempt+1}/{max_retries}), retry dans {wait}s...")
                time.sleep(wait)
                continue
            # Dernière tentative échouée ou erreur non-réseau
            raise RuntimeError(
                f"Ollama inaccessible après {attempt+1} tentative(s). "
                f"Cause : {err[:200]}. "
                f"Solution : vérifiez 'ollama serve' et 'ollama list'."
            ) from e


# ─────────────────────────────────────────────────────────────
# MODÈLE DE SIMULATION (sans Ollama)
# ─────────────────────────────────────────────────────────────
class MockViticulureModel:
    """Modèle de simulation — répond sans avoir besoin d'Ollama."""

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        last = ""
        if isinstance(messages, list):
            for m in reversed(messages):
                content = getattr(m, 'content', None) or (
                    m[-1].get('content', '') if isinstance(m, list) else ''
                )
                if content and isinstance(content, str):
                    last = content.lower()
                    break
        return self._respond(last)

    def _respond(self, query: str) -> AIMessage:
        if any(k in query for k in ["mildiou", "tache huileuse", "tache oil"]):
            return AIMessage(content="""**Analyse : Suspicion Mildiou** (Plasmopara viticola)

**Symptômes caractéristiques :**
- Taches huileuses translucides (face supérieure des feuilles)
- Duvet blanc cotonneux en face inférieure (sporulation)
- Brunissement des inflorescences

**Conditions favorables :** T° > 13°C + pluie > 10mm + HR > 75%

**Recommandations immédiates :**
1. Traitement préventif cuprique (bouillie bordelaise 150-300g Cu/ha)
2. Rognage pour aérer le feuillage
3. Contrôle visuel toutes les 48h

Précisez le stade phénologique pour affiner le diagnostic.""")

        if any(k in query for k in ["oïdium", "blanc", "farineux", "poudre"]):
            return AIMessage(content="""**Analyse : Suspicion Oïdium** (Erysiphe necator)

**Symptômes :** Feutrage blanc farineux sur feuilles, rameaux et grappes.

**Traitement :**
- Soufre mouillable : 2-3 kg/ha (préventif)
- IBE (tébuconazole) si attaque déclarée
- DAR soufre : 5 jours

Quel est le cépage et la date de floraison ?""")

        if any(k in query for k in ["diagnostic complet", "rapport", "analyse complète", "diagnostic structuré"]):
            return AIMessage(content="""Diagnostic complet en cours...

J'analyse les symptômes décrits et les conditions météo pour générer le rapport structuré.

Le diagnostic JSON sera disponible dans la réponse structurée.""")

        return AIMessage(content="""Je suis VITI-AI, votre assistant agronome viticole.

Pour un diagnostic précis, indiquez-moi :
1. 🌿 **Stade phénologique** (débourrement, floraison, véraison...)
2. 📍 **Localisation** (région, appellation)
3. 🍇 **Cépage(s)**
4. 👁️ **Symptômes observés** (organes atteints, aspect, couleur)
5. 🌦️ **Météo récente** (pluie, température, humidité)

Je dispose d'outils météo, de bases de données phytosanitaires et de calculateurs agronomiques.""")


# ─────────────────────────────────────────────────────────────
# NŒUDS DU GRAPHE
# ─────────────────────────────────────────────────────────────
def node_agent(state: AgentState, model) -> AgentState:
    """Nœud principal : appelle le LLM avec retry et troncature."""
    context = state.get("parcelle_context", {}) or {}

    messages_formatted = PROMPT_AGENT.format_messages(
        parcelle_context=json.dumps(context, ensure_ascii=False) if context else "Non renseigné",
        date_actuelle=datetime.now().strftime("%d/%m/%Y %H:%M"),
        messages=state["messages"],
    )

    try:
        response = invoke_with_retry(model, messages_formatted)
    except RuntimeError as e:
        response = AIMessage(content=(
            f"⚠️ **Connexion Ollama perdue.**\n\n{e}\n\n"
            "En attendant, voici ce que je peux vous dire :\n"
            "- Vérifiez `ollama serve` dans un terminal\n"
            "- Vérifiez `ollama list` pour confirmer le modèle installé\n"
            "- Relancez votre requête dans quelques secondes"
        ))

    return {"messages": [response]}


def node_tools(state: AgentState) -> AgentState:
    """Nœud d'exécution des outils."""
    tool_node = ToolNode(VIGNE_TOOLS)
    return tool_node.invoke(state)


def node_extract_diagnostic(state: AgentState, model) -> AgentState:
    """
    Nœud d'extraction du diagnostic structuré.
    LCEL : PROMPT_DIAGNOSTIC | model_runnable | JsonOutputParser
    """
    # Résumé court de la conversation (max 1500 chars pour rester dans le contexte)
    conv_lines = []
    for m in state["messages"]:
        if hasattr(m, 'content') and isinstance(m.content, str) and m.content.strip():
            role = type(m).__name__.replace("Message", "")
            conv_lines.append(f"{role}: {m.content[:150]}")
    conv_text = "\n".join(conv_lines[-4:])  # 4 messages max pour llama3.1 8B

    # Météo depuis les ToolMessages
    meteo_context = "Non disponible"
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage) and "risques_phytosanitaires" in str(msg.content):
            meteo_context = str(msg.content)[:400]
            break

    # LCEL : prompt | model_runnable | JsonOutputParser
    parser = JsonOutputParser()

    def model_invoke_safe(prompt_value):
        msgs = prompt_value.to_messages() if hasattr(prompt_value, 'to_messages') else prompt_value
        try:
            return invoke_with_retry(model, msgs, max_retries=1)
        except Exception as e:
            # Retourne un JSON minimal si le modèle échoue
            return AIMessage(content=json.dumps({
                "stade_phenologique": "feuillaison",
                "maladies_detectees": [{"nom": "À déterminer", "nom_scientifique": None,
                                        "probabilite": 0.5, "symptomes_observes": [],
                                        "organes_touches": []}],
                "severite_globale": "moderée", "risque_propagation": 0.5,
                "traitements_recommandes": [], "actions_culturales": [],
                "surveillance": "Inspection visuelle régulière",
                "resume_diagnostic": f"Diagnostic partiel (erreur LLM : {str(e)[:80]})",
                "alerte": False, "confidence_score": 0.3
            }))

    model_runnable = RunnableLambda(model_invoke_safe)
    chain = PROMPT_DIAGNOSTIC | model_runnable | parser

    try:
        diagnostic_data = chain.invoke({
            "conversation": conv_text[:600],   # Max 600 chars → stable avec modèles 8B
            "meteo_context": meteo_context[:200],
        })
        # Validation Pydantic
        diagnostic_obj = DiagnosticVigne(**diagnostic_data)
        diagnostic_dict = diagnostic_obj.model_dump()
    except Exception as e:
        diagnostic_dict = _diagnostic_fallback(conv_text, str(e))

    return {
        "diagnostic": diagnostic_dict,
        "messages": [AIMessage(
            content=(
                f"✅ **Diagnostic structuré généré.**\n\n"
                f"**Résumé :** {diagnostic_dict.get('resume_diagnostic', 'Voir rapport')}\n"
                f"**Sévérité :** {diagnostic_dict.get('severite_globale', 'N/A')}\n"
                f"**Alerte urgente :** {'🔴 OUI — Intervention immédiate' if diagnostic_dict.get('alerte') else '🟢 NON'}\n"
                f"**Confiance :** {int(diagnostic_dict.get('confidence_score', 0)*100)}%\n\n"
                f"Le rapport JSON complet est disponible via GET `/diagnostic/{{thread_id}}`."
            )
        )],
    }


def _diagnostic_fallback(conversation: str, error: str) -> dict:
    """Diagnostic de repli par détection de mots-clés."""
    maladies = []
    low = conversation.lower()
    if "mildiou" in low or "tache huileuse" in low:
        maladies.append({"nom": "Mildiou", "nom_scientifique": "Plasmopara viticola",
                         "probabilite": 0.75, "symptomes_observes": ["Taches huileuses"],
                         "organes_touches": ["Feuilles", "Grappes"]})
    if "oïdium" in low or "blanc" in low or "farineux" in low:
        maladies.append({"nom": "Oïdium", "nom_scientifique": "Erysiphe necator",
                         "probabilite": 0.70, "symptomes_observes": ["Feutrage blanc"],
                         "organes_touches": ["Feuilles", "Grappes"]})
    if not maladies:
        maladies.append({"nom": "Non déterminé", "nom_scientifique": None,
                         "probabilite": 0.4, "symptomes_observes": ["À préciser"],
                         "organes_touches": ["À examiner"]})
    return {
        "parcelle_id": None, "cepage": None,
        "stade_phenologique": "feuillaison",
        "date_diagnostic": datetime.now().strftime("%Y-%m-%d"),
        "maladies_detectees": maladies,
        "severite_globale": "moderée",
        "risque_propagation": 0.5,
        "traitements_recommandes": [{
            "produit": "Bouillie bordelaise",
            "type_traitement": "préventif",
            "dose": "150-300g Cu/ha",
            "timing": "Dès que conditions météo favorables",
            "precautions": ["Porter des EPI"],
            "delai_avant_recolte": None,
        }],
        "actions_culturales": ["Améliorer l'aération", "Surveiller 2×/semaine"],
        "surveillance": "Contrôle visuel tous les 3-4 jours",
        "resume_diagnostic": "Diagnostic préliminaire (analyse conversationnelle). Inspection visuelle recommandée.",
        "alerte": False,
        "confidence_score": 0.45,
    }


def route_after_agent(state: AgentState) -> str:
    """Routage conditionnel après le nœud agent."""
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    last_human = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_human = m.content.lower()
            break

    if any(kw in last_human for kw in [
        "diagnostic complet", "rapport complet", "analyse complète",
        "diagnostic structuré", "génère un rapport", "bilan complet"
    ]):
        return "extract_diagnostic"

    return END


# ─────────────────────────────────────────────────────────────
# CONSTRUCTION DU GRAPHE LANGGRAPH
# ─────────────────────────────────────────────────────────────
def build_vigne_agent(
    use_ollama: bool = True,
    model_name: str = "qwen3.5:4b",
    timeout: int = 180,
    num_ctx: int = 2048,
    num_predict: int = 512,
    db_path: str = "viti_ai_conversations.db",
):
    """Compile le graphe LangGraph avec SqliteSaver (persistance entre redémarrages)."""
    base_model = get_model(use_ollama, model_name, timeout, num_ctx, num_predict)

    model_with_tools = (
        base_model.bind_tools(VIGNE_TOOLS)
        if hasattr(base_model, 'bind_tools')
        else base_model
    )

    import sqlite3 as _sqlite3
    _conn = _sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(_conn)
    checkpointer.setup()           # Crée les tables si elles n'existent pas
    print(f"\U0001f4be SqliteSaver actif \u2192 {db_path}")

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", lambda s: node_agent(s, model_with_tools))
    workflow.add_node("tools", node_tools)
    workflow.add_node("extract_diagnostic", lambda s: node_extract_diagnostic(s, base_model))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", route_after_agent,
                                   {"tools": "tools",
                                    "extract_diagnostic": "extract_diagnostic",
                                    END: END})
    workflow.add_edge("tools", "agent")
    workflow.add_edge("extract_diagnostic", END)

    return workflow.compile(checkpointer=checkpointer)



# ─────────────────────────────────────────────────────────────
# INTERFACE DE CONVERSATION
# ─────────────────────────────────────────────────────────────
def chat_avec_agent(
    app,
    message: str,
    thread_id: str = "session_001",
    parcelle_context: Optional[dict] = None,
) -> dict:
    """Envoie un message et retourne la réponse (mémoire via thread_id)."""
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "messages": [HumanMessage(content=message)],
        "thread_id": thread_id,
        "mode": "chat",
        "diagnostic": None,
        "parcelle_context": parcelle_context or {},
    }

    result = app.invoke(initial_state, config=config)

    last_ai = next(
        (m.content for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
        "Aucune réponse générée."
    )

    return {
        "response": last_ai,
        "diagnostic": result.get("diagnostic"),
        "thread_id": thread_id,
        "messages_count": len(result["messages"]),
    }