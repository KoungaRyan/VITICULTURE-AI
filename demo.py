"""
Démonstration complète de l'Agent Agronome Viticole VITI-AI
============================================================
Ce script teste toutes les composantes du système :
  1. Outils individuels (météo, seuils, calculs)
  2. Pipeline LCEL (prompt | model | parser)
  3. Agent conversationnel avec mémoire
  4. Extraction du diagnostic structuré Pydantic
"""

import json
import sys
sys.path.insert(0, '/home/claude/vigne_agent')

from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


def separateur(titre: str):
    print(f"\n{'═'*60}")
    print(f"  {titre}")
    print(f"{'═'*60}")


# ─────────────────────────────────────────────────────────────
# ÉTAPE 1 : Test des Outils
# ─────────────────────────────────────────────────────────────
def demo_outils():
    separateur("ÉTAPE 1 — Outils Agronomes")
    
    from tools.vigne_tools import get_meteo_vigne, get_seuils_alerte, calcul_agronomique
    
    # 1a. Outil Météo — Bordeaux
    print("\n📡 [Outil 1] Météo pour Bordeaux (44.83°N, -0.57°E) :")
    meteo = get_meteo_vigne.invoke({"latitude": 44.83, "longitude": -0.57, "jours": 3})
    data = json.loads(meteo)
    # Gestion du fallback si l'API externe est indisponible
    actuel = data.get('actuel') or data.get('donnees_defaut', {})
    risques = data.get('risques_phytosanitaires') or actuel.get('risques_phytosanitaires', {})
    source = "(données par défaut - API hors réseau)" if 'erreur' in data else "(Open-Meteo)"
    print(f"  Source : {source}")
    print(f"  Température : {actuel.get('temperature_c', 'N/A')}°C")
    print(f"  Humidité    : {actuel.get('humidite_pct', 'N/A')}%")
    print(f"  Risque Mildiou : {risques.get('mildiou', 'N/A')}")
    print(f"  Risque Oïdium  : {risques.get('oidium', 'N/A')}")
    print(f"  Risque Botrytis: {risques.get('botrytis', 'N/A')}")
    
    # 1b. Outil Seuils — Mildiou en floraison
    print("\n📊 [Outil 2] Seuils d'alerte — Mildiou au stade floraison :")
    seuils = get_seuils_alerte.invoke({"maladie": "mildiou", "stade_vigne": "floraison"})
    data_s = json.loads(seuils)
    print(f"  Stade sensible : {'⚠️ OUI' if data_s.get('alerte_stade') else 'Non'}")
    print(f"  Seuil traitement : {data_s.get('seuil_traitement', 'N/A')[:80]}...")
    symptomes = data_s.get('symptomes_cles', [])
    for s in symptomes[:2]:
        print(f"  • {s}")
    
    # 1c. Calculs agronomiques
    print("\n🔢 [Outil 3] Calculs agronomiques :")
    
    # Degrés-jours
    dj = calcul_agronomique.invoke({"operation": "degres_jours", "temp_base": 10, "temp_moy_jour": 22, "jours": 1})
    dj_data = json.loads(dj)
    print(f"  Degrés-Jours (T_moy=22°C, T_base=10°C) : {dj_data['resultat']} °C.j")
    print(f"  Interprétation : {dj_data['interpretation']}")
    
    # Dose traitement
    dose = calcul_agronomique.invoke({"operation": "dose_traitement", "surface_ha": 3.5, "dose_L_ha": 2.5})
    dose_data = json.loads(dose)
    print(f"\n  Dose pour 3.5 ha @ 2.5 L/ha : {dose_data['volume_total_L']} L")
    print(f"  {dose_data['conseil']}")
    
    # Potentiel alcool
    pa = calcul_agronomique.invoke({"operation": "potentiel_alcool", "densite": 1082})
    pa_data = json.loads(pa)
    print(f"\n  Potentiel alcool (densité 1082) : {pa_data['potentiel_alcool_pct_vol']}% vol.")
    print(f"  {pa_data['interpretation']}")
    
    # Indice mildiou
    im = calcul_agronomique.invoke({
        "operation": "indice_mildiou", 
        "temperature": 20, "pluie_mm": 8, "humidite_pct": 85
    })
    im_data = json.loads(im)
    print(f"\n  Indice Mildiou EPI (T=20°C, pluie=8mm, HR=85%) : {im_data['indice_EPI']}")
    print(f"  Risque : {im_data['risque']}")


# ─────────────────────────────────────────────────────────────
# ÉTAPE 2 : Pipeline LCEL
# ─────────────────────────────────────────────────────────────
def demo_lcel():
    separateur("ÉTAPE 2 — Pipeline LCEL (prompt | model | parser)")

    from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough

    print("\n🔗 [LCEL] Démonstration des pipelines : prompt | model | parser")

    prompt_agro = ChatPromptTemplate.from_messages([
        ("system", "Expert viticole. Reponds UNIQUEMENT en JSON : {{\"maladie\":\"nom\",\"urgence\":\"niveau\",\"action\":\"str\"}}"),
        ("human", "{question}")
    ])

    def simule_llm(prompt_value):
        msgs = prompt_value.to_messages()
        content = str(msgs[-1].content) if msgs else ""
        if "mildiou" in content.lower() or "tache" in content.lower():
            return AIMessage(content='{"maladie":"Mildiou (Plasmopara viticola)","urgence":"elevee","action":"Traitement cuprique immediat"}')

    llm = RunnableLambda(simule_llm)

    # Pipeline 1 : prompt | llm | StrOutputParser
    chain_str = prompt_agro | llm | StrOutputParser()
    r1 = chain_str.invoke({"question": "Taches huileuses sur feuilles de vigne, mildiou ?"})
    print(f"\n  Pipeline 1 (str)  → {r1[:70]}")

    # Pipeline 2 : prompt | llm | JsonOutputParser
    chain_json = prompt_agro | llm | JsonOutputParser()
    r2 = chain_json.invoke({"question": "Taches huileuses mildiou vigne"})
    print(f"  Pipeline 2 (json) → maladie={r2.get('maladie')} | urgence={r2.get('urgence')}")

    # Pipeline 3 : RunnablePassthrough + enrichissement
    chain3 = (
        RunnablePassthrough.assign(question=lambda x: x["question"])
        | prompt_agro | llm | JsonOutputParser()
        | RunnableLambda(lambda d: {**d, "source": "VITI-AI v1.0", "confiance": 0.85})
    )
    r3 = chain3.invoke({"question": "Taches huileuses mildiou"})
    print(f"  Pipeline 3 (enrich) → source={r3.get('source')} | confiance={r3.get('confiance')}")

    print("\n✅ LCEL Pipelines fonctionnels : prompt → model → parser → enrichissement")


# ─────────────────────────────────────────────────────────────
# ÉTAPE 3 : Agent Conversationnel avec Mémoire
# ─────────────────────────────────────────────────────────────
def demo_agent_memoire():
    separateur("ÉTAPE 3 — Agent Conversationnel + Mémoire (thread_id)")
    
    from agent.vigne_agent import build_vigne_agent, chat_avec_agent
    
    print("\n🏗️  Construction du graphe LangGraph...")
    agent = build_vigne_agent(use_ollama=True, model_name="llama3.1")
    print("✅ Agent compilé avec MemorySaver (checkpointer)")
    
    # Session de conversation — thread_id fixe pour la mémoire
    thread_id = "vigneron_dupont_parcelle_A"
    
    parcelle_ctx = {
        "nom_vigneron": "M. Dupont",
        "localisation": "Saint-Émilion, Gironde",
        "cepage_principal": "Merlot",
        "superficie_ha": 4.5,
        "altitude_m": 85,
        "stade_actuel": "floraison"
    }
    
    print(f"\n📋 Contexte parcelle : {parcelle_ctx['localisation']}, {parcelle_ctx['cepage_principal']}, {parcelle_ctx['superficie_ha']} ha")
    print(f"   Thread ID : {thread_id}")
    
    # Tour 1 : Présentation du problème
    print("\n" + "─"*50)
    print("👨‍🌾 Message 1 : Présentation du problème")
    result1 = chat_avec_agent(
        agent,
        "Bonjour, je suis vigneron à Saint-Émilion. Depuis hier, je remarque des taches huileuses sur mes feuilles de Merlot. Les grappes sont au stade floraison.",
        thread_id=thread_id,
        parcelle_context=parcelle_ctx
    )
    print(f"\n🤖 VITI-AI :")
    response_lines = result1['response'].split('\n') if result1['response'] else ['(Pas de réponse)']
    for line in response_lines[:8]:
        if line.strip():
            print(f"   {line}")
    if len(response_lines) > 8:
        print("   [...]")
    
    # Tour 2 : Question de suivi (utilise la mémoire)
    print("\n" + "─"*50)
    print("👨‍🌾 Message 2 : Question de suivi (teste la mémoire)")
    result2 = chat_avec_agent(
        agent,
        "Il a plu 12mm avant-hier et la température est à 21°C avec 80% d'humidité. Est-ce dangereux ?",
        thread_id=thread_id,
        parcelle_context=parcelle_ctx
    )
    print(f"\n🤖 VITI-AI :")
    response2_lines = result2['response'].split('\n') if result2['response'] else ['(Pas de réponse)']
    for line in response2_lines[:8]:
        if line.strip():
            print(f"   {line}")
    if len(response2_lines) > 8:
        print("   [...]")
    
    print(f"\n📊 Mémoire : {result2['messages_count']} messages dans l'historique (thread: {thread_id})")
    
    return agent, thread_id, parcelle_ctx


# ─────────────────────────────────────────────────────────────
# ÉTAPE 4 : Diagnostic Structuré (Output Parser + Pydantic)
# ─────────────────────────────────────────────────────────────
def demo_diagnostic_structure(agent, thread_id, parcelle_ctx):
    separateur("ÉTAPE 4 — Diagnostic Structuré (Output Parser + Pydantic)")
    
    from agent.vigne_agent import chat_avec_agent
    from models.schemas import DiagnosticVigne
    
    print("\n🔬 Demande de diagnostic complet structuré...")
    
    result = chat_avec_agent(
        agent,
        "Faites un diagnostic complet structuré de ma vigne en tenant compte de tout ce que je vous ai dit.",
        thread_id=thread_id,
        parcelle_context=parcelle_ctx
    )
    
    print(f"\n🤖 VITI-AI : {result['response']}")
    
    if result.get("diagnostic"):
        diag = result["diagnostic"]
        print(f"\n{'─'*50}")
        print("📋 DIAGNOSTIC STRUCTURÉ (JSON Pydantic) :")
        print(f"{'─'*50}")
        print(f"  Stade phénologique : {diag.get('stade_phenologique', 'N/A')}")
        print(f"  Sévérité globale   : {diag.get('severite_globale', 'N/A').upper()}")
        print(f"  Alerte urgente     : {'🔴 OUI' if diag.get('alerte') else '🟢 NON'}")
        print(f"  Score confiance    : {diag.get('confidence_score', 0)*100:.0f}%")
        print(f"  Risque propagation : {diag.get('risque_propagation', 0)*100:.0f}%")
        
        maladies = diag.get("maladies_detectees", [])
        if maladies:
            print(f"\n  🦠 Maladies détectées ({len(maladies)}) :")
            for m in maladies:
                print(f"    • {m['nom']}")
                if m.get('nom_scientifique'):
                    print(f"      ({m['nom_scientifique']})")
                print(f"      Probabilité : {m['probabilite']*100:.0f}%")
                for s in m.get('symptomes_observes', [])[:2]:
                    print(f"      - {s}")
        
        traitements = diag.get("traitements_recommandes", [])
        if traitements:
            print(f"\n  💊 Traitements recommandés ({len(traitements)}) :")
            for t in traitements[:2]:
                print(f"    • {t['produit']} ({t['type_traitement']})")
                print(f"      Timing : {t['timing']}")
                if t.get('dose'):
                    print(f"      Dose : {t['dose']}")
                if t.get('delai_avant_recolte'):
                    print(f"      DAR : {t['delai_avant_recolte']} jours")
        
        actions = diag.get("actions_culturales", [])
        if actions:
            print(f"\n  🌿 Actions culturales :")
            for a in actions[:3]:
                print(f"    • {a}")
        
        print(f"\n  📝 Résumé : {diag.get('resume_diagnostic', 'N/A')}")
        print(f"  👁️  Surveillance : {diag.get('surveillance', 'N/A')}")
        
        # Validation Pydantic
        print(f"\n{'─'*50}")
        print("✅ Validation Pydantic DiagnosticVigne :")
        try:
            diag_obj = DiagnosticVigne(**diag)
            print(f"   ✓ Schéma validé avec succès")
            print(f"   ✓ Type: {type(diag_obj).__name__}")
            print(f"   ✓ Champs: {len(diag_obj.model_fields)} champs définis")
        except Exception as e:
            print(f"   ⚠️  Validation partielle : {e}")
    else:
        print("\n  ℹ️  Diagnostic structuré non généré dans cette session de démo.")
        print("  (Normal en mode simulation — le modèle réel Ollama génère le JSON)")


# ─────────────────────────────────────────────────────────────
# ÉTAPE 5 : Affichage de l'Architecture
# ─────────────────────────────────────────────────────────────
def affiche_architecture():
    separateur("ÉTAPE 5 — Architecture Complète du Système")
    
    print("""
┌─────────────────────────────────────────────────────────────┐
│              VITI-AI — Agent Agronome Viticole              │
│           Système Multi-Agent Détection Maladies            │
└─────────────────────────────────────────────────────────────┘

PIPELINE LCEL :
  ChatPromptTemplate ──► ChatOllama (llama3.2) ──► JsonOutputParser
         │                       │                        │
    Système agrono.          Tools binding          DiagnosticVigne
    Variables context       (ReAct pattern)          (Pydantic v2)

GRAPHE LANGGRAPH :
                     ┌─────────────┐
          START ────►│    AGENT    │
                     │ (llama3.2 +│
                     │   tools)   │
                     └──────┬──────┘
                            │
              ┌─────────────┼──────────────┐
              ▼             ▼              ▼
         ┌─────────┐  ┌──────────┐      END
         │  TOOLS  │  │EXTRACT   │
         │  Node   │  │DIAGNOSTIC│
         │meteo    │  │LCEL chain│
         │seuils   │  │→ Pydantic│
         │calcul   │  └────┬─────┘
         └────┬────┘       │
              │            ▼
              └──► AGENT   END
                   (retry)

MÉMOIRE (MemorySaver) :
  thread_id="vigneron_dupont_A" ──► [msg1, msg2, msg3, ...]
  thread_id="domaine_bordeaux_B" ──► [msg1, msg2, ...]
  (Chaque thread conserve son historique indépendant)

OUTILS DISPONIBLES :
  1. get_meteo_vigne(lat, lon, jours)
     └─ Open-Meteo API → conditions + risques (mildiou/oïdium/botrytis)
  
  2. get_seuils_alerte(maladie, stade_vigne)  
     └─ Base de données phytosanitaire → seuils officiels + produits
  
  3. calcul_agronomique(operation, **params)
     └─ degres_jours | dose_traitement | potentiel_alcool | 
        indice_mildiou | surface_feuillaire

SCHÉMA DIAGNOSTIC (DiagnosticVigne — Pydantic) :
  ├── parcelle_id, cepage, stade_phenologique
  ├── maladies_detectees[] → {nom, probabilite, symptomes, organes}
  ├── severite_globale: faible|moderée|élevée|critique
  ├── risque_propagation: 0.0–1.0
  ├── traitements_recommandes[] → {produit, type, dose, timing, DAR}
  ├── actions_culturales[]
  ├── surveillance, resume_diagnostic
  ├── alerte: bool
  └── confidence_score: 0.0–1.0

API REST (FastAPI) :
  POST /chat              → Conversation + mémoire thread_id
  GET  /diagnostic/{id}  → Récupération diagnostic structuré
  POST /tools/meteo      → Météo + risques phytosanitaires
  POST /tools/seuils     → Seuils d'alerte maladies
  POST /tools/calcul     → Calculs agronomiques
  GET  /schema/diagnostic → Schéma JSON DiagnosticVigne
    """)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🌿 VITI-AI — Démonstration Agent Agronome Viticole")
    print(f"   {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    
    # Étape 1 : Outils
    demo_outils()
    
    # Étape 2 : LCEL
    demo_lcel()
    
    # Étape 3 : Agent avec mémoire  
    agent, thread_id, parcelle_ctx = demo_agent_memoire()
    
    # Étape 4 : Diagnostic structuré
    demo_diagnostic_structure(agent, thread_id, parcelle_ctx)
    
    # Étape 5 : Architecture
    affiche_architecture()
    
    print("\n" + "═"*60)
    print("  ✅ Démonstration complète terminée avec succès !")
    print("  📚 Pour démarrer l'API : uvicorn api.main:app --reload")
    print("  🦙 Pour Ollama : ollama pull llama3.2")
    print("═"*60 + "\n")
