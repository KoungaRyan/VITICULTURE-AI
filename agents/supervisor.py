"""
Supervisor — orchestrateur multi-agent VITI-AI
================================================
Remplace l'ancien agent monolithique (agent/vigne_agent.py, 6 tools bindés
sur un seul LLM) par un routeur qui délègue à 3 agents spécialisés, chacun
connecté à son propre serveur MCP (agents/weather_agent.py, image_agent.py,
text_rag_agent.py). Le Supervisor ne voit jamais les tools bas niveau : il
voit 3 (+1) "consulter_agent_..." et décide qui appeler.

Conserve le même comportement observable que l'ancien agent :
  - mémoire persistante par thread_id (SqliteSaver)
  - extraction d'un diagnostic structuré Pydantic sur demande explicite
  - même interface chat_avec_agent(app, message, thread_id, parcelle_context)
    → l'API (api/main.py) n'a donc qu'un import à changer.
"""
import json
from typing import Annotated, Sequence, TypedDict, Optional
from datetime import datetime

from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode

from agent.vigne_agent import get_model, invoke_with_retry  # réutilisés tels quels
from agents.weather_agent import weather_agent
from agents.image_agent import image_agent
from agents.text_rag_agent import text_rag_agent
from tools.vigne_tools import get_contexte_parcelle
from models.schemas import DiagnosticVigne


# ─────────────────────────────────────────────────────────────
# ÉTAT DU GRAPHE (identique à l'ancien agent — compat API)
# ─────────────────────────────────────────────────────────────
class SupervisorState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    thread_id: str
    mode: str
    diagnostic: Optional[dict]
    parcelle_context: Optional[dict]


# ─────────────────────────────────────────────────────────────
# AGENTS SPÉCIALISÉS VUS COMME DES TOOLS PAR LE SUPERVISOR
# ─────────────────────────────────────────────────────────────
@tool
def consulter_agent_meteo(question: str) -> str:
    """
    Délègue à l'agent Météo/Agronomie : conditions météo, risques
    phytosanitaires (mildiou/oïdium/botrytis), seuils d'alerte officiels,
    calculs agronomiques (degrés-jours, dose, potentiel alcool, EPI, LAI).
    Formule la question en langage naturel, avec les coordonnées/paramètres
    nécessaires (ex: "météo et risque mildiou à Bordeaux 44.83,-0.57").
    """
    result = weather_agent.invoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].content


@tool
def consulter_agent_image(question: str) -> str:
    """
    Délègue à l'agent Image RAG : diagnostic d'une maladie à partir d'une
    photo, par similarité vectorielle avec les images de référence déjà
    labellisées. Précise dans la question le chemin/URL de l'image et
    l'id_parcelle. Ne PAS appeler sans image réelle disponible.
    """
    result = image_agent.invoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].content


@tool
def consulter_agent_rag_texte(question: str) -> str:
    """
    Délègue à l'agent RAG Texte : recherche dans la base documentaire
    phytosanitaire (guides, fiches maladies, réglementation) pour justifier
    une recommandation avec une source précise (posologie, DAR, symptômes).
    """
    result = text_rag_agent.invoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].content


SUPERVISOR_TOOLS = [
    consulter_agent_meteo,
    consulter_agent_image,
    consulter_agent_rag_texte,
    get_contexte_parcelle,  # gardé côté Supervisor : sert de contexte à toutes les délégations
]


# ─────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────
SYSTEM_SUPERVISOR = """Tu es VITI-AI, l'orchestrateur d'un système multi-agent viticole.
Date: {date_actuelle}
Parcelle: {parcelle_context}
Maladies gérées: mildiou, oïdium, botrytis, black-rot, excoriose, cochylis, acariens.

Tu ne réponds JAMAIS toi-même sur la météo, une image, ou de la
documentation technique : tu DÉLÈGUES systématiquement à l'agent spécialisé
compétent (consulter_agent_meteo / consulter_agent_image /
consulter_agent_rag_texte). Utilise get_contexte_parcelle pour vérifier
l'historique avant de conclure. Synthétise ensuite les réponses des agents
pour l'utilisateur, en français, de façon concise (max 150 mots). Rappelle
les DAR. Propose des alternatives bio."""

PROMPT_SUPERVISOR = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_SUPERVISOR),
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
    ("human", "Conversation :\n{conversation}\n\nContexte agents :\n{agents_context}"),
])


# ─────────────────────────────────────────────────────────────
# NŒUD SUPERVISOR
# ─────────────────────────────────────────────────────────────
def node_supervisor(state: SupervisorState, model_with_tools) -> dict:
    prompt_value = PROMPT_SUPERVISOR.invoke({
        "messages": state["messages"],
        "date_actuelle": datetime.now().strftime("%Y-%m-%d"),
        "parcelle_context": json.dumps(state.get("parcelle_context") or {}, ensure_ascii=False),
    })
    response = invoke_with_retry(model_with_tools, prompt_value.to_messages())
    return {"messages": [response]}


def route_after_supervisor(state: SupervisorState) -> str:
    """Routage conditionnel après le nœud supervisor (identique dans l'esprit
    à route_after_agent de l'ancien vigne_agent.py)."""
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


def node_extract_diagnostic(state: SupervisorState, model) -> dict:
    conv_text = "\n".join(
        f"{'Utilisateur' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}"
        for m in state["messages"] if isinstance(m, (HumanMessage, AIMessage)) and m.content
    )

    agents_context = "Non disponible"
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            agents_context = str(msg.content)[:400]
            break

    parser = JsonOutputParser()

    def model_invoke_safe(prompt_value):
        msgs = prompt_value.to_messages() if hasattr(prompt_value, "to_messages") else prompt_value
        try:
            return invoke_with_retry(model, msgs, max_retries=1)
        except Exception as e:
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
            "conversation": conv_text[:600],
            "agents_context": agents_context[:200],
        })
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


# ─────────────────────────────────────────────────────────────
# CONSTRUCTION DU GRAPHE
# ─────────────────────────────────────────────────────────────
def build_supervisor(
    use_ollama: bool = True,
    model_name: str = "llama3.2:3b",
    timeout: int = 180,
    num_ctx: int = 2048,
    num_predict: int = 512,
    db_path: str = "data/viti_ai_conversations.db",
):
    """Compile le graphe Supervisor (remplace build_vigne_agent)."""
    base_model = get_model(use_ollama, model_name, timeout, num_ctx, num_predict)

    model_with_tools = (
        base_model.bind_tools(SUPERVISOR_TOOLS)
        if hasattr(base_model, "bind_tools")
        else base_model
    )

    import sqlite3 as _sqlite3
    _conn = _sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(_conn)
    checkpointer.setup()

    workflow = StateGraph(SupervisorState)

    workflow.add_node("supervisor", lambda s: node_supervisor(s, model_with_tools))
    workflow.add_node("tools", ToolNode(SUPERVISOR_TOOLS))
    workflow.add_node("extract_diagnostic", lambda s: node_extract_diagnostic(s, base_model))

    workflow.set_entry_point("supervisor")
    workflow.add_conditional_edges("supervisor", route_after_supervisor, {
        "tools": "tools",
        "extract_diagnostic": "extract_diagnostic",
        END: END,
    })
    workflow.add_edge("tools", "supervisor")
    workflow.add_edge("extract_diagnostic", END)

    return workflow.compile(checkpointer=checkpointer)


# ─────────────────────────────────────────────────────────────
# INTERFACE DE CONVERSATION — même signature que chat_avec_agent
# ─────────────────────────────────────────────────────────────
def chat_avec_supervisor(
    app,
    message: str,
    thread_id: str = "session_001",
    parcelle_context: Optional[dict] = None,
) -> dict:
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
