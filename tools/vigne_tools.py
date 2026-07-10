"""
Outils (Tools) de l'agent agronome viticole.
Étape 2 : Calcul, météo, seuils d'alerte,accès à l'historique structuré du Data Commons .
"""
from __future__ import annotations
import json
import requests
from langchain_core.tools import tool
from data_commons.storage import DataCommonsStore
from vector_store.milvus_client import VectorStore
from embeddings.text_embedder import embed_text
from collections import Counter
from embeddings.image_embedder import embed_image

SEUIL_CONFIANCE_MIN = 0.55  # score de similarité cosinus en dessous duquel on ne conclut pas

_vector_store = VectorStore()
_data_commons_store = DataCommonsStore()

# ─────────────────────────────────────────
# OUTIL 1 : Données météo via Open-Meteo
# ─────────────────────────────────────────
@tool
def get_meteo_vigne(latitude: float, longitude: float, jours: int = 3) -> str:
    """
    Récupère les données météo actuelles et prévisions pour évaluer
    les risques de maladies (mildiou, oïdium, botrytis) sur la vigne.

    Args:
        latitude: Latitude de la parcelle (ex: 44.83 pour Bordeaux)
        longitude: Longitude de la parcelle (ex: -0.57 pour Bordeaux)
        jours: Nombre de jours de prévision (1 à 7)

    Returns:
        JSON avec conditions météo et indices de risque maladies
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": [
                "temperature_2m_max", "temperature_2m_min",
                "precipitation_sum", "windspeed_10m_max",
                "relative_humidity_2m_max", "relative_humidity_2m_min"
            ],
            "current": [
                "temperature_2m", "relative_humidity_2m",
                "precipitation", "windspeed_10m"
            ],
            "forecast_days": min(jours, 7),
            "timezone": "Europe/Paris"
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()

        current = data.get("current", {})
        daily = data.get("daily", {})

        temp = current.get("temperature_2m", 18)
        humidity = current.get("relative_humidity_2m", 70)
        precip = current.get("precipitation", 0)

        # ── Calcul indices de risque ──────────────────────────────
        # Mildiou (Plasmopara viticola) : chaud + humide + pluie
        risque_mildiou = _calcul_risque_mildiou(temp, humidity, precip)
        # Oïdium (Erysiphe necator) : chaud, sec à modéré, 20–28°C
        risque_oïdium = _calcul_risque_oidium(temp, humidity)
        # Botrytis : humidité très élevée, T° douce
        risque_botrytis = _calcul_risque_botrytis(temp, humidity, precip)

        result = {
            "source": "Open-Meteo",
            "coordonnees": {"lat": latitude, "lon": longitude},
            "actuel": {
                "temperature_c": temp,
                "humidite_pct": humidity,
                "precipitation_mm": precip,
                "vent_kmh": current.get("windspeed_10m", 0)
            },
            "risques_phytosanitaires": {
                "mildiou": risque_mildiou,
                "oidium": risque_oïdium,
                "botrytis": risque_botrytis
            },
            "previsions_j1_j3": {
                "temp_max": daily.get("temperature_2m_max", [])[:3],
                "temp_min": daily.get("temperature_2m_min", [])[:3],
                "precipitation_mm": daily.get("precipitation_sum", [])[:3],
                "humidite_max": daily.get("relative_humidity_2m_max", [])[:3]
            }
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "erreur": f"Impossible de récupérer la météo : {str(e)}",
            "donnees_defaut": {
                "temperature_c": 18,
                "humidite_pct": 70,
                "risques_phytosanitaires": {
                    "mildiou": "Modéré",
                    "oidium": "Faible",
                    "botrytis": "Faible"
                }
            }
        })


def _calcul_risque_mildiou(temp: float, humidity: float, precip: float) -> str:
    score = 0
    if 13 <= temp <= 30:
        score += 2
    if humidity >= 75:
        score += 2
    if precip > 5:
        score += 3
    if precip > 0.3 and humidity >= 90:
        score += 2  # règle des 3-10 (Müller)
    if score >= 6:
        return "Élevé"
    elif score >= 3:
        return "Modéré"
    return "Faible"


def _calcul_risque_oidium(temp: float, humidity: float) -> str:
    score = 0
    if 20 <= temp <= 28:
        score += 3
    elif 15 <= temp < 20:
        score += 1
    if 45 <= humidity <= 75:
        score += 2
    elif humidity > 75:
        score += 1
    if score >= 4:
        return "Élevé"
    elif score >= 2:
        return "Modéré"
    return "Faible"


def _calcul_risque_botrytis(temp: float, humidity: float, precip: float) -> str:
    score = 0
    if 15 <= temp <= 25:
        score += 2
    if humidity >= 85:
        score += 3
    if precip > 2:
        score += 2
    if score >= 5:
        return "Élevé"
    elif score >= 3:
        return "Modéré"
    return "Faible"


# ─────────────────────────────────────────
# OUTIL 2 : Seuils d'alerte phytosanitaires
# ─────────────────────────────────────────
@tool
def get_seuils_alerte(maladie: str, stade_vigne: str) -> str:
    """
    Retourne les seuils officiels d'alerte et de traitement pour une
    maladie ou ravageur de la vigne selon le stade phénologique.

    Args:
        maladie: Nom de la maladie (mildiou, oïdium, botrytis, excoriose,
                 black-rot, cicadelle, eudémis, cochylis, acariens, vers_grise)
        stade_vigne: Stade phénologique (débourrement, feuillaison, floraison,
                     nouaison, véraison, maturité)

    Returns:
        JSON avec seuils, symptômes et fenêtre de traitement
    """
    seuils_db = {
        "mildiou": {
            "nom_scientifique": "Plasmopara viticola",
            "conditions_infection": {
                "temperature_min": 13,
                "temperature_max": 30,
                "humidite_min": 75,
                "pluie_declencheur_mm": 0.3,
                "periode_incubation_jours": "4-18 selon température"
            },
            "stades_sensibles": ["feuillaison", "floraison", "nouaison"],
            "seuil_traitement": "Règle des 3-10 : grappes atteignent 10 cm + T°>13°C + 10mm de pluie en 24-48h",
            "organes_touches": ["feuilles (taches huileuses)", "grappes", "vrilles", "pousses"],
            "symptomes_cles": [
                "Taches huileuses jaune-vert sur feuilles (face supérieure)",
                "Duvet blanc cotonneux face inférieure (fructification sporangifère)",
                "Brunissement et dessèchement des inflorescences",
                "Grains bruns et ratatinés (rot brun)"
            ],
            "produits_homologues": [
                {"nom": "Cuivre (Bouillie Bordeaux)", "type": "préventif", "dose": "150-300g Cu/ha"},
                {"nom": "Fosetyl-Al (Aliette)", "type": "systémique", "dose": "2.5 kg/ha"},
                {"nom": "Mancozèbe", "type": "de contact", "dose": "1.5-2 kg/ha"}
            ]
        },
        "oïdium": {
            "nom_scientifique": "Erysiphe necator (Uncinula necator)",
            "conditions_infection": {
                "temperature_optimale": "20-28°C",
                "humidite": "Ne nécessite pas de pluie, spores actives à HR > 45%",
                "lumiere": "Freiné par exposition directe UV prolongée"
            },
            "stades_sensibles": ["débourrement", "feuillaison", "floraison", "nouaison"],
            "seuil_traitement": "Dès l'apparition des premiers symptômes ou prophylactiquement au débourrement",
            "organes_touches": ["feuilles", "grappes", "jeunes pousses"],
            "symptomes_cles": [
                "Feutrage blanc grisâtre farineux sur feuilles et grappes",
                "Feuilles crispées, tordues (attaque précoce)",
                "Éclatement des baies (attaque sur jeunes grains)",
                "Odeur caractéristique de moisi"
            ],
            "produits_homologues": [
                {"nom": "Soufre mouillable", "type": "préventif/curatif", "dose": "2-3 kg/ha"},
                {"nom": "Trifloxystrobine (Flint)", "type": "systémique", "dose": "75-100 g/ha"},
                {"nom": "Tébuconazole", "type": "IBE (curatif)", "dose": "500 ml/ha"}
            ]
        },
        "botrytis": {
            "nom_scientifique": "Botrytis cinerea",
            "conditions_infection": {
                "temperature_optimale": "15-25°C",
                "humidite_min": 85,
                "facteurs_aggravants": ["blessures baies", "densité feuillage", "vendange tardive"]
            },
            "stades_sensibles": ["floraison", "fermeture grappe", "véraison", "maturité"],
            "seuil_traitement": "Traitement préventif obligatoire à la fermeture de la grappe si conditions favorables",
            "organes_touches": ["grappes (pourriture grise)", "pédoncules", "feuilles (rare)"],
            "symptomes_cles": [
                "Moisissure grise duveteuse sur les baies",
                "Pourriture brune avec feutrage gris (sporulation)",
                "Pédoncules et rafles bruns et desséchés",
                "Odeur caractéristique (acide acétique)"
            ],
            "produits_homologues": [
                {"nom": "Fenhexamide (Teldor)", "type": "curatif", "dose": "1 kg/ha", "DAR": 7},
                {"nom": "Boscalide + Pyraclostrobine (Bellis)", "type": "préventif/curatif", "dose": "1 kg/ha"},
                {"nom": "Fludioxonil (Switch)", "type": "de contact", "dose": "0.8 kg/ha"}
            ]
        },
        "black-rot": {
            "nom_scientifique": "Guignardia bidwellii",
            "stades_sensibles": ["feuillaison", "floraison", "nouaison"],
            "seuil_traitement": "Traitement à la sortie des feuilles si historique positif sur parcelle",
            "symptomes_cles": [
                "Taches brunes circulaires avec halo jaune sur feuilles",
                "Pycnides noires disposées en cercle sur les taches",
                "Grains momifiés noirs (mumies)"
            ]
        },
        "excoriose": {
            "nom_scientifique": "Phomopsis viticola",
            "stades_sensibles": ["débourrement", "feuillaison"],
            "seuil_traitement": "Traitement obligatoire au débourrement (1-2 feuilles étalées) si historique",
            "symptomes_cles": [
                "Taches décolorées à la base des sarments",
                "Nécroses noires sur entre-noeuds (1er à 4ème)",
                "Déformation en Z des sarments"
            ]
        },
        "cochylis": {
            "nom_scientifique": "Eupoecilia ambiguella",
            "stades_sensibles": ["floraison (G1)", "fermeture grappe (G2)"],
            "seuil_traitement": "Piégeage : > 15-20 adultes/piège/semaine (G1) ; dès l'observation de larves (G2)",
            "symptomes_cles": [
                "Larves roses-rougeâtres dans les boutons floraux liés en glomérules",
                "Baies rongées et parasitées (G2)",
                "Entrée de pourriture secondaire"
            ]
        },
        "acariens": {
            "especes": ["Panonychus ulmi (araignée rouge)", "Calepitrimerus vitis (acariose)"],
            "stades_sensibles": ["débourrement", "feuillaison"],
            "seuil_traitement": "Araignée rouge : > 50% feuilles infestées + > 5 femelles/feuille en juillet",
            "symptomes_cles": [
                "Bronzage et plissement des feuilles (acariose)",
                "Décoloration bronzée face supérieure (araignée rouge)",
                "Présence de toiles sur feuilles"
            ]
        }
    }

    maladie_lower = maladie.lower().replace("ï", "ï")
    
    # Recherche approximative
    for key in seuils_db:
        if key in maladie_lower or maladie_lower in key:
            info = seuils_db[key].copy()
            info["stade_demande"] = stade_vigne
            
            # Alerte si stade sensible
            stades_sensibles = info.get("stades_sensibles", [])
            info["alerte_stade"] = any(
                s.lower() in stade_vigne.lower() for s in stades_sensibles
            )
            return json.dumps(info, ensure_ascii=False, indent=2)

    return json.dumps({
        "message": f"Maladie '{maladie}' non trouvée dans la base.",
        "maladies_disponibles": list(seuils_db.keys())
    }, ensure_ascii=False)


# ─────────────────────────────────────────
# OUTIL 3 : Calculs agronomiques
# ─────────────────────────────────────────
@tool
def calcul_agronomique(operation: str, **kwargs) -> str:
    """
    Effectue des calculs agronomiques pour la viticulture.

    Args:
        operation: Type de calcul parmi :
            - 'degres_jours' : Calcul des degrés-jours de croissance
            - 'dose_traitement' : Calcul dose de produit selon surface et concentration
            - 'indice_mildiou' : Indice EPI (Effective Primary Inoculum)
            - 'potentiel_alcool' : Estimation degré alcoolique depuis densité mout
            - 'surface_feuillaire' : Estimation surface foliaire
        **kwargs: Paramètres spécifiques à chaque calcul

    Returns:
        JSON avec résultat et explication du calcul

    Exemples d'appel:
        calcul_agronomique("degres_jours", temp_base=10, temp_moy_jour=22)
        calcul_agronomique("dose_traitement", surface_ha=3.5, dose_L_ha=2.5)
        calcul_agronomique("potentiel_alcool", densite=1085)
    """
    try:
        if operation == "degres_jours":
            temp_base = float(kwargs.get("temp_base", 10))
            temp_moy = float(kwargs.get("temp_moy_jour", 20))
            jours = int(kwargs.get("jours", 1))
            dj = max(0, temp_moy - temp_base) * jours
            return json.dumps({
                "operation": "Degrés-Jours de Croissance (DJC)",
                "formule": f"max(0, T_moy - T_base) × jours = max(0, {temp_moy} - {temp_base}) × {jours}",
                "resultat": round(dj, 2),
                "unite": "°C.jours",
                "interpretation": _interprete_dj(dj * jours if jours > 1 else dj)
            }, ensure_ascii=False)

        elif operation == "dose_traitement":
            surface = float(kwargs.get("surface_ha", 1))
            dose_ha = float(kwargs.get("dose_L_ha", 2))
            concentration = float(kwargs.get("concentration_pct", 100))
            volume_total = surface * dose_ha * (concentration / 100)
            return json.dumps({
                "operation": "Calcul de dose produit",
                "surface_ha": surface,
                "dose_recommandee_ha": f"{dose_ha} L/ha",
                "concentration": f"{concentration}%",
                "volume_total_L": round(volume_total, 2),
                "volume_bouillie_necessaire": f"{round(surface * dose_ha, 2)} L de bouillie",
                "conseil": "Vérifier le volume d'eau du pulvérisateur (150-300 L/ha vigne)"
            }, ensure_ascii=False)

        elif operation == "potentiel_alcool":
            densite = float(kwargs.get("densite", 1080))
            # Formule Brix approximative depuis densité
            brix = (densite - 1000) / 3.95  
            # Potentiel alcool (formule empirique)
            potentiel = brix * 0.59  
            return json.dumps({
                "operation": "Potentiel alcoolique",
                "densite_moût": densite,
                "brix_estime": round(brix, 1),
                "potentiel_alcool_pct_vol": round(potentiel, 1),
                "interpretation": _interprete_potentiel(potentiel),
                "formule": "Brix ≈ (densité - 1000) / 3.95 ; Alcool ≈ Brix × 0.59"
            }, ensure_ascii=False)

        elif operation == "indice_mildiou":
            # Modèle simplifié EPI
            temp = float(kwargs.get("temperature", 18))
            pluie = float(kwargs.get("pluie_mm", 5))
            humidite = float(kwargs.get("humidite_pct", 70))
            
            facteur_t = max(0, 1 - abs(temp - 22) / 12)  # Optimum à 22°C
            facteur_h = max(0, (humidite - 60) / 40)
            facteur_p = min(1, pluie / 10)
            epi = round(facteur_t * facteur_h * facteur_p * 100, 1)
            
            return json.dumps({
                "operation": "Indice de risque Mildiou (EPI simplifié)",
                "parametres": {"temperature": temp, "pluie_mm": pluie, "humidite_pct": humidite},
                "indice_EPI": epi,
                "risque": "Élevé ⚠️" if epi > 60 else "Modéré" if epi > 30 else "Faible",
                "conseil": "Traitement préventif recommandé si EPI > 60 en stade sensible"
            }, ensure_ascii=False)

        elif operation == "surface_feuillaire":
            longeur_cm = float(kwargs.get("longueur_feuille_cm", 15))
            largeur_cm = float(kwargs.get("largeur_feuille_cm", 14))
            nb_feuilles = int(kwargs.get("nb_feuilles_par_rameau", 15))
            nb_rameaux = int(kwargs.get("nb_rameaux_par_pied", 8))
            nb_pieds_ha = int(kwargs.get("nb_pieds_ha", 5000))
            
            sf_feuille = longeur_cm * largeur_cm * 0.78  # facteur de forme
            sf_rameau = sf_feuille * nb_feuilles
            sf_pied = sf_rameau * nb_rameaux
            sf_ha = (sf_pied * nb_pieds_ha) / 10000  # en m²
            
            return json.dumps({
                "operation": "Surface Foliaire Exposée",
                "sf_par_feuille_cm2": round(sf_feuille, 1),
                "sf_par_pied_m2": round(sf_pied / 10000, 2),
                "sf_totale_ha_m2": round(sf_ha, 0),
                "indice_surf_foliaire": round(sf_ha / 10000, 2),  # LAI
                "interpretation": "LAI optimal vigne : 1.5–2.5 m²/m²"
            }, ensure_ascii=False)

        else:
            return json.dumps({
                "erreur": f"Opération '{operation}' inconnue.",
                "operations_disponibles": [
                    "degres_jours", "dose_traitement", "potentiel_alcool",
                    "indice_mildiou", "surface_feuillaire"
                ]
            }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"erreur": str(e)}, ensure_ascii=False)


def _interprete_dj(dj):
    if dj < 200:
        return "Début de végétation (débourrement à feuillaison)"
    elif dj < 600:
        return "Pleine croissance végétative"
    elif dj < 1000:
        return "Floraison à nouaison"
    elif dj < 1400:
        return "Grossissement des baies à véraison"
    else:
        return "Maturation - récolte imminente"


def _interprete_potentiel(pa):
    if pa < 10:
        return "Vendange prématurée - attendre"
    elif pa < 12:
        return "Maturité insuffisante pour vinification standard"
    elif pa <= 14:
        return "Maturité correcte ✓"
    elif pa <= 15.5:
        return "Bonne maturité, surveiller l'évolution"
    else:
        return "Sur-maturité - risque de déséquilibre"



@tool
def get_contexte_parcelle(id_parcelle: str, limite: int = 10) -> dict:
    """
    Retourne l'historique météo/maturation/maladies/traitements d'une parcelle.

    À utiliser AVANT de produire un diagnostic ou une recommandation de
    traitement, pour vérifier si un traitement a déjà été appliqué récemment,
    si une maladie a déjà été observée sur cette parcelle, ou pour comparer
    la maturation actuelle à la tendance récente.

    Args:
        id_parcelle: identifiant de la parcelle (ex: "irouleguy-pilote-01")
        limite: nombre max d'enregistrements par catégorie à retourner (défaut 10)

    Returns:
        dict avec les clés: meteo, maturation, maladies, traitements
    """
    return _data_commons_store.get_historique_parcelle(id_parcelle, limite)



@tool
def rechercher_connaissance_phytosanitaire(question: str, categorie: str | None = None, top_k: int = 3) -> dict:
    """
    Recherche dans la base documentaire phytosanitaire (guides de traitement,
    fiches maladies, réglementation) pour répondre à une question technique
    ou justifier une recommandation.

    Utiliser ce tool quand une réponse nécessite une justification technique
    précise (posologie, délai avant récolte, symptômes caractéristiques) qui
    ne se trouve ni dans get_seuils_alerte (seuils chiffrés) ni dans le
    Data Commons structuré (historique).

    Args:
        question: la question ou le sujet à rechercher
        categorie: filtre optionnel ("maladie", "traitement", "reglementation")
        top_k: nombre d'extraits à retourner (défaut 3)

    Returns:
        dict avec une liste "extraits" (texte, source, score_similarite)
    """
    vecteur_requete = embed_text(question, task_prefix="search_query")
    resultats = _vector_store.search_connaissances(vecteur_requete, top_k=top_k, categorie=categorie)

    return {
        "question": question,
        "extraits": resultats,
        "nb_resultats": len(resultats),
    }



@tool
def diagnostiquer_image_maladie(image_path: str, id_parcelle: str, top_k: int = 5) -> dict:
    """
    Diagnostique une maladie de la vigne à partir d'une image (photo robot/drone)
    en la comparant par similarité vectorielle aux images de référence déjà
    labellisées dans le Data Commons.

    Utiliser ce tool quand une image est disponible (après une prise de vue
    robot déclenchée pour lever un doute, par exemple). Ne PAS l'utiliser sans
    image réelle — combiner ensuite avec get_seuils_alerte et
    get_contexte_parcelle pour la recommandation finale.

    Args:
        image_path: chemin local ou URL de l'image à diagnostiquer
        id_parcelle: parcelle d'où provient l'image (pour tracer le résultat)
        top_k: nombre d'images de référence à comparer (défaut 5)

    Returns:
        dict: maladie_probable, confiance (0-1), severite_estimee,
              nb_references_consultees, references (détail des matches),
              alerte_faible_confiance (bool)
    """
    vecteur = embed_image(image_path)
    matches = _vector_store.search_images_similaires(vecteur, top_k=top_k)

    if not matches:
        return {
            "maladie_probable": None,
            "confiance": 0.0,
            "alerte_faible_confiance": True,
            "message": "Aucune image de référence dans le Data Commons pour comparaison.",
        }

    # Vote pondéré par similarité (cosinus, Milvus renvoie déjà la distance/score)
    votes = Counter()
    for m in matches:
        votes[m["maladie"]] += m["score_similarite"]

    maladie_probable, score_total = votes.most_common(1)[0]
    confiance = score_total / sum(votes.values())

    severites_associees = [m["severite"] for m in matches if m["maladie"] == maladie_probable]
    severite_estimee = round(sum(severites_associees) / len(severites_associees), 1)

    return {
        "maladie_probable": maladie_probable,
        "confiance": round(float(confiance), 3),
        "severite_estimee": severite_estimee,
        "nb_references_consultees": len(matches),
        "references": matches,
        "alerte_faible_confiance": confiance < SEUIL_CONFIANCE_MIN,
        "id_parcelle": id_parcelle,
    }


# Liste des outils disponibles
VIGNE_TOOLS = [get_meteo_vigne, get_seuils_alerte, calcul_agronomique, get_contexte_parcelle,rechercher_connaissance_phytosanitaire,diagnostiquer_image_maladie]
