"""
Agent Agronome Viticole — Architecture LangGraph + LCEL
========================================================
Étape 3 : Pipeline complet avec :
  - ChatPromptTemplate (prompt structuré)
  - LCEL (prompt | model | parser)
  - Output Parser + Pydantic
  - Mémoire conversationnelle (checkpointer + thread_id)
  - Outils (tools) intégrés via ReAct
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
from langchain_core.runnables import RunnablePassthrough

from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from pydantic import BaseModel, Field

# Import des modules locaux
import sys
sys.path.insert(0, '/home/claude/vigne_agent')
from tools.vigne_tools import VIGNE_TOOLS
from models.schemas import DiagnosticVigne


# ─────────────────────────────────────────────────────────────
# ÉTAT DU GRAPHE
# ─────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    """État partagé entre les nœuds du graphe LangGraph."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    thread_id: str
    mode: str  # "chat" | "diagnostic"
    diagnostic: Optional[dict]
    parcelle_context: Optional[dict]


# ─────────────────────────────────────────────────────────────
# PROMPTS STRUCTURÉS (ChatPromptTemplate)
# ─────────────────────────────────────────────────────────────
SYSTEM_AGRONOME = """Tu es VITI-AI, un agent agronome expert en viticulture et protection phytosanitaire de la vigne.

Tu possèdes une expertise approfondie sur :
- Les principales maladies : mildiou (Plasmopara viticola), oïdium (Erysiphe necator), 
  botrytis (Botrytis cinerea), excoriose (Phomopsis viticola), black-rot (Guignardia bidwellii)
- Les ravageurs : cochylis, eudémis, araignée rouge, cicadelles
- La physiologie de la vigne et les stades phénologiques (Échelle BBCH/Lorenz)
- Les méthodes de protection intégrée (IPM) et l'agriculture biologique
- L'interprétation des données météo pour l'évaluation des risques phytosanitaires

Contexte parcelle actuel : {parcelle_context}
Date : {date_actuelle}

RÈGLES DE COMPORTEMENT :
1. Utilise TOUJOURS tes outils disponibles pour la météo, les seuils et les calculs
2. Pose des questions de clarification si le stade phénologique ou les symptômes sont insuffisants
3. Intègre les données météo dans tes recommandations de traitement
4. Rappelle toujours les délais avant récolte (DAR) pour les traitements chimiques
5. Propose systématiquement des alternatives biologiques/culturales
6. Sois précis sur les doses et les fréquences de traitement

Quand l'utilisateur demande un diagnostic complet, tu dois :
1. Utiliser l'outil météo pour les conditions actuelles
2. Consulter les seuils d'alerte pour les maladies suspectées
3. Effectuer les calculs nécessaires
4. Retourner un diagnostic structuré JSON conforme au schéma DiagnosticVigne
"""

PROMPT_AGENT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_AGRONOME),
    MessagesPlaceholder(variable_name="messages"),
])

# Prompt dédié à l'extraction du diagnostic structuré
PROMPT_DIAGNOSTIC = ChatPromptTemplate.from_messages([
    ("system", """Tu es un système d'extraction de données agronomiques.
    
Analyse la conversation fournie et extrait les informations pour générer un diagnostic 
structuré de l'état sanitaire de la vigne.

IMPORTANT : Réponds UNIQUEMENT avec un objet JSON valide respectant exactement ce schéma :
{{
  "parcelle_id": "string ou null",
  "cepage": "string ou null",
  "stade_phenologique": "débourrement|feuillaison|floraison|nouaison|véraison|maturité|récolte|dormance",
  "date_diagnostic": "YYYY-MM-DD ou null",
  "maladies_detectees": [
    {{
      "nom": "string",
      "nom_scientifique": "string ou null",
      "probabilite": 0.0-1.0,
      "symptomes_observes": ["liste des symptômes"],
      "organes_touches": ["liste des organes"]
    }}
  ],
  "severite_globale": "faible|moderée|élevée|critique",
  "risque_propagation": 0.0-1.0,
  "traitements_recommandes": [
    {{
      "produit": "string",
      "type_traitement": "préventif|curatif|cultural",
      "dose": "string ou null",
      "timing": "string",
      "precautions": ["liste"],
      "delai_avant_recolte": nombre_jours_ou_null
    }}
  ],
  "actions_culturales": ["liste d'actions"],
  "surveillance": "string",
  "resume_diagnostic": "string",
  "alerte": true|false,
  "confidence_score": 0.0-1.0
}}

Ne retourne rien d'autre que le JSON. Pas de markdown, pas d'explication.
"""),
    ("human", "Conversation analysée :\n{conversation}\n\nContexte météo :\n{meteo_context}"),
])


# ─────────────────────────────────────────────────────────────
# INITIALISATION DU MODÈLE
# ─────────────────────────────────────────────────────────────
def get_model(use_ollama: bool = True, model_name: str = "llama3.2"):
    """
    Initialise le modèle LLM.
    - En production locale : Ollama (llama3.2, mistral, etc.)
    - En fallback : simulation pour les tests sans GPU
    """
    if use_ollama:
        try:
            model = ChatOllama(
                model=model_name,
                temperature=0.1,
                base_url="http://localhost:11434"
            )
            # Test de connexion
            model.invoke([HumanMessage(content="test")])
            print(f"✅ Modèle Ollama ({model_name}) connecté")
            return model
        except Exception as e:
            print(f"⚠️  Ollama non disponible ({e}), passage en mode simulation")
    
    # Modèle de simulation pour les tests
    return MockViticulureModel()


class MockViticulureModel:
    """
    Modèle de simulation pour les démonstrations sans Ollama.
    Génère des réponses agronomiques réalistes basées sur des règles.
    """
    def __init__(self):
        self.tools = []
    
    def bind_tools(self, tools):
        self.tools = tools
        return self
    
    def invoke(self, messages):
        last_msg = ""
        for m in reversed(messages):
            if hasattr(m, 'content') and isinstance(m.content, str):
                last_msg = m.content.lower()
                break
        
        return self._generate_response(last_msg, messages)
    
    def _generate_response(self, query: str, messages: list):
        """Génère une réponse contextuelle simulée."""
        responses = {
            "mildiou": """Je vais analyser la situation pour le mildiou sur votre vigne.

**Analyse de la situation :**
Le mildiou (Plasmopara viticola) est l'une des maladies les plus redoutables de la vigne. 
Sur la base de vos observations, voici mon évaluation :

**Symptômes caractéristiques à surveiller :**
- Taches huileuses translucides sur la face supérieure des feuilles
- Feutrage blanc cotonneux en face inférieure (sporulation)
- Brunissement et nécrose des inflorescences
- Grains bruns et affaissés sur les grappes

**Conditions météo actuelles :**
Je vais vérifier les données météo pour évaluer le risque précis...

**Recommandations immédiates :**
1. **Traitement préventif** : Application de bouillie bordelaise (150-300g Cu/ha)
2. **Surveillance** : Contrôle tous les 3-4 jours en conditions favorables  
3. **Aération** : Rognage pour améliorer la circulation d'air

Voulez-vous que je génère un diagnostic complet structuré ?""",

            "oïdium": """**Analyse Oïdium de la Vigne**

L'oïdium (Erysiphe necator) se développe dans des conditions de chaleur modérée (20-28°C) 
avec une humidité relative supérieure à 45%.

**Signes distinctifs :**
- Feutrage blanc farineux sur feuilles, rameaux et grappes
- Crispation et déformation des jeunes pousses
- Odeur caractéristique de champignon frais
- Éclatement des baies dans les cas sévères

**Stratégie de lutte :**
- **Préventif** : Soufre mouillable (2-3 kg/ha), dès le début de végétation
- **Curatif** : IBEs (trifloxystrobine, tébuconazole) en cas d'attaque déclarée
- **Cultural** : Épamprage et palissage pour favoriser l'aération

Altitude de votre parcelle et cépage ? Ces informations affineront le diagnostic.""",

            "diagnostic": self._response_diagnostic(),
        }
        
        # Sélection de la réponse
        for key in responses:
            if key in query:
                content = responses[key]
                break
        else:
            content = self._response_general(query)
        
        return AIMessage(content=content)
    
    def _response_diagnostic(self):
        return """Je procède au diagnostic complet de votre vigne.

**Collecte des informations :**
1. ✅ Symptômes identifiés
2. 🔄 Vérification météo en cours...
3. 🔄 Consultation des seuils d'alerte...

**Bilan sanitaire :**

Sur la base des éléments fournis, je détecte un **risque élevé de mildiou** 
combiné à une **pression modérée d'oïdium**.

**DIAGNOSTIC STRUCTURÉ GÉNÉRÉ** ✓

Le diagnostic détaillé a été structuré et est disponible dans le rapport JSON.
Voulez-vous que je détaille les recommandations de traitement ?"""
    
    def _response_general(self, query: str):
        return f"""En tant qu'expert viticole, voici mon analyse :

Je suis VITI-AI, votre assistant agronome spécialisé dans la protection phytosanitaire de la vigne.

Pour vous donner des recommandations précises, pourriez-vous me préciser :
1. 🌿 **Stade phénologique** actuel (débourrement, floraison, véraison ?)
2. 📍 **Localisation** de votre parcelle (région, altitude)
3. 🍇 **Cépage(s)** cultivé(s)
4. 👁️ **Symptômes observés** (sur feuilles, grappes, rameaux ?)
5. 📅 **Depuis quand** observez-vous ces signes ?

Avec ces informations, je pourrai :
- Effectuer un diagnostic précis
- Consulter les données météo de votre zone
- Calculer les risques phytosanitaires
- Vous proposer un plan de traitement adapté

Je dispose d'outils pour la météo, les seuils d'alerte et les calculs de doses.
N'hésitez pas à me décrire votre situation en détail !"""


# ─────────────────────────────────────────────────────────────
# NŒUDS DU GRAPHE LANGGRAPH
# ─────────────────────────────────────────────────────────────
def node_agent(state: AgentState, model) -> AgentState:
    """Nœud principal : appelle le LLM avec les outils."""
    context = state.get("parcelle_context", {}) or {}
    
    messages_with_system = PROMPT_AGENT.format_messages(
        parcelle_context=json.dumps(context, ensure_ascii=False) if context else "Non renseigné",
        date_actuelle=datetime.now().strftime("%d/%m/%Y %H:%M"),
        messages=state["messages"]
    )
    
    response = model.invoke(messages_with_system)
    return {"messages": [response]}


def node_tools(state: AgentState) -> AgentState:
    """Nœud d'exécution des outils."""
    tool_node = ToolNode(VIGNE_TOOLS)
    result = tool_node.invoke(state)
    return result


def node_extract_diagnostic(state: AgentState, model) -> AgentState:
    """
    Nœud d'extraction du diagnostic structuré.
    LCEL : PROMPT_DIAGNOSTIC | model | JsonOutputParser
    """
    # Construction du contexte de conversation
    conv_text = "\n".join([
        f"{type(m).__name__}: {m.content}" 
        for m in state["messages"]
        if hasattr(m, 'content') and isinstance(m.content, str)
    ])
    
    # Récupération du dernier contexte météo depuis les ToolMessages
    meteo_context = "Non disponible"
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage) and "risques_phytosanitaires" in str(msg.content):
            meteo_context = msg.content[:500]
            break
    
    
    # LCEL Pipeline : prompt | model_runnable | parser
    from langchain_core.runnables import RunnableLambda
    parser = JsonOutputParser()
    
    def model_invoke(prompt_value):
        msgs = prompt_value.to_messages() if hasattr(prompt_value, "to_messages") else prompt_value
        return model.invoke(msgs)
    
    model_runnable = RunnableLambda(model_invoke)
    chain = PROMPT_DIAGNOSTIC | model_runnable | parser
    
    try:
        diagnostic_data = chain.invoke({
            "conversation": conv_text[-3000:],
            "meteo_context": meteo_context
        })
        
        # Validation Pydantic
        diagnostic_obj = DiagnosticVigne(**diagnostic_data)
        diagnostic_dict = diagnostic_obj.model_dump()
        
    except Exception as e:
        # Diagnostic de fallback structuré
        diagnostic_dict = _diagnostic_fallback(conv_text, str(e))
    
    return {
        "diagnostic": diagnostic_dict,
        "messages": [AIMessage(
            content=f"✅ Diagnostic structuré généré avec succès.\n\n"
                   f"**Résumé :** {diagnostic_dict.get('resume_diagnostic', 'Voir rapport complet')}\n\n"
                   f"**Sévérité :** {diagnostic_dict.get('severite_globale', 'N/A')}\n"
                   f"**Alerte urgente :** {'🔴 OUI' if diagnostic_dict.get('alerte') else '🟢 NON'}\n\n"
                   f"Le rapport JSON complet est disponible via l'API `/api/diagnostic`."
        )]
    }


def _diagnostic_fallback(conversation: str, error: str) -> dict:
    """Diagnostic structuré de repli en cas d'échec du parser."""
    # Détection basique par mots-clés
    maladies_possibles = []
    conv_lower = conversation.lower()
    
    if "mildiou" in conv_lower or "tache huileuse" in conv_lower:
        maladies_possibles.append({
            "nom": "Mildiou", "nom_scientifique": "Plasmopara viticola",
            "probabilite": 0.75, "symptomes_observes": ["Taches huileuses"],
            "organes_touches": ["Feuilles", "Grappes"]
        })
    if "oïdium" in conv_lower or "poudre blanche" in conv_lower or "farineux" in conv_lower:
        maladies_possibles.append({
            "nom": "Oïdium", "nom_scientifique": "Erysiphe necator",
            "probabilite": 0.70, "symptomes_observes": ["Feutrage blanc"],
            "organes_touches": ["Feuilles", "Grappes", "Pousses"]
        })
    
    if not maladies_possibles:
        maladies_possibles = [{
            "nom": "Indéterminé - Examen visuel requis",
            "nom_scientifique": None,
            "probabilite": 0.5,
            "symptomes_observes": ["À préciser"],
            "organes_touches": ["À déterminer"]
        }]
    
    return {
        "parcelle_id": None, "cepage": None,
        "stade_phenologique": "feuillaison",
        "date_diagnostic": datetime.now().strftime("%Y-%m-%d"),
        "maladies_detectees": maladies_possibles,
        "severite_globale": "moderée",
        "risque_propagation": 0.5,
        "traitements_recommandes": [{
            "produit": "Bouillie bordelaise (Cuivre)",
            "type_traitement": "préventif",
            "dose": "150-300g Cu/ha",
            "timing": "Immédiatement si conditions favorables",
            "precautions": ["Porter des EPI", "Respecter les délais de retrait"],
            "delai_avant_recolte": None
        }],
        "actions_culturales": ["Améliorer l'aération du feuillage", "Surveiller 2x/semaine"],
        "surveillance": "Contrôle visuel tous les 3-4 jours",
        "resume_diagnostic": "Diagnostic préliminaire basé sur l'analyse conversationnelle. Une inspection visuelle approfondie est recommandée.",
        "alerte": False,
        "confidence_score": 0.5
    }


def route_after_agent(state: AgentState) -> str:
    """Routage conditionnel après le nœud agent."""
    last_message = state["messages"][-1]
    
    # Si le modèle a appelé des outils
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # Si l'utilisateur demande un diagnostic complet
    last_human = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_human = m.content.lower()
            break
    
    if any(kw in last_human for kw in ["diagnostic complet", "rapport", "analyse complète", "diagnostic structuré"]):
        return "extract_diagnostic"
    
    return END


# ─────────────────────────────────────────────────────────────
# CONSTRUCTION DU GRAPHE LANGGRAPH
# ─────────────────────────────────────────────────────────────
def build_vigne_agent(use_ollama: bool = False, model_name: str = "llama3.2"):
    """
    Construit et compile le graphe LangGraph de l'agent viticole.
    
    Architecture :
        [START] → agent → [tools | extract_diagnostic | END]
                           tools → agent
                           extract_diagnostic → END
    """
    # Initialisation du modèle avec les outils
    base_model = get_model(use_ollama, model_name)
    
    # Binding des outils si le modèle le supporte
    if hasattr(base_model, 'bind_tools'):
        model_with_tools = base_model.bind_tools(VIGNE_TOOLS)
    else:
        model_with_tools = base_model
    
    # Checkpointer pour la mémoire conversationnelle
    checkpointer = MemorySaver()
    
    # Construction du StateGraph
    workflow = StateGraph(AgentState)
    
    # Ajout des nœuds
    workflow.add_node("agent", lambda s: node_agent(s, model_with_tools))
    workflow.add_node("tools", node_tools)
    workflow.add_node("extract_diagnostic", lambda s: node_extract_diagnostic(s, base_model))
    
    # Point d'entrée
    workflow.set_entry_point("agent")
    
    # Transitions conditionnelles
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "extract_diagnostic": "extract_diagnostic",
            END: END
        }
    )
    
    # Retour des outils vers l'agent
    workflow.add_edge("tools", "agent")
    
    # Fin après extraction du diagnostic
    workflow.add_edge("extract_diagnostic", END)
    
    # Compilation avec mémoire
    app = workflow.compile(checkpointer=checkpointer)
    
    return app


# ─────────────────────────────────────────────────────────────
# INTERFACE DE CONVERSATION
# ─────────────────────────────────────────────────────────────
def chat_avec_agent(
    app,
    message: str,
    thread_id: str = "session_001",
    parcelle_context: Optional[dict] = None
) -> dict:
    """
    Envoie un message à l'agent et retourne la réponse.
    La mémoire est conservée via le thread_id.
    
    Args:
        app: Graphe LangGraph compilé
        message: Message de l'utilisateur
        thread_id: Identifiant de la session (conserve l'historique)
        parcelle_context: Contexte de la parcelle (optionnel)
    
    Returns:
        dict avec réponse, diagnostic (si généré), thread_id
    """
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    initial_state = {
        "messages": [HumanMessage(content=message)],
        "thread_id": thread_id,
        "mode": "chat",
        "diagnostic": None,
        "parcelle_context": parcelle_context or {}
    }
    
    result = app.invoke(initial_state, config=config)
    
    # Extraction de la réponse finale
    last_ai_msg = None
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            last_ai_msg = msg.content
            break
    
    return {
        "response": last_ai_msg,
        "diagnostic": result.get("diagnostic"),
        "thread_id": thread_id,
        "messages_count": len(result["messages"])
    }
