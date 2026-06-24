"""
Schémas Pydantic pour le diagnostic structuré de la vigne.
Étape 1 : Modèles de données pour l'Output Parser.
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class SeveriteEnum(str, Enum):
    FAIBLE = "faible"
    MODEREE = "moderée"
    ELEVEE = "élevée"
    CRITIQUE = "critique"


class StadeVigne(str, Enum):
    DORMANCE = "dormance"
    DEBOURREMENT = "débourrement"
    FEUILLAISON = "feuillaison"
    FLORAISON = "floraison"
    NOUAISON = "nouaison"
    VERAISON = "véraison"
    MATURITE = "maturité"
    RECOLTE = "récolte"


class Maladie(BaseModel):
    """Maladie ou ravageur détecté."""
    nom: str = Field(description="Nom de la maladie ou du ravageur")
    nom_scientifique: Optional[str] = Field(None, description="Nom scientifique")
    probabilite: float = Field(ge=0.0, le=1.0, description="Probabilité de 0 à 1")
    symptomes_observes: List[str] = Field(description="Symptômes visibles correspondants")
    organes_touches: List[str] = Field(description="Organes de la vigne affectés")


class TraitementRecommande(BaseModel):
    """Traitement recommandé pour la vigne."""
    produit: str = Field(description="Produit ou méthode de traitement")
    type_traitement: str = Field(description="Préventif, curatif ou cultural")
    dose: Optional[str] = Field(None, description="Dose recommandée")
    timing: str = Field(description="Moment d'application")
    precautions: List[str] = Field(default_factory=list, description="Précautions d'emploi")
    delai_avant_recolte: Optional[int] = Field(None, description="DAR en jours")


class DiagnosticVigne(BaseModel):
    """
    Diagnostic complet et structuré de l'état sanitaire de la vigne.
    Schéma principal retourné par l'Output Parser.
    """
    # Contexte
    parcelle_id: Optional[str] = Field(None, description="Identifiant de la parcelle")
    cepage: Optional[str] = Field(None, description="Cépage concerné")
    stade_phenologique: StadeVigne = Field(description="Stade de la vigne au moment du diagnostic")
    date_diagnostic: Optional[str] = Field(None, description="Date du diagnostic")

    # Analyse sanitaire
    maladies_detectees: List[Maladie] = Field(
        description="Liste des maladies ou ravageurs identifiés"
    )
    severite_globale: SeveriteEnum = Field(description="Sévérité globale de l'atteinte")
    risque_propagation: float = Field(
        ge=0.0, le=1.0,
        description="Risque de propagation rapide (0=faible, 1=certain)"
    )

    # Recommandations
    traitements_recommandes: List[TraitementRecommande] = Field(
        description="Traitements à effectuer par ordre de priorité"
    )
    actions_culturales: List[str] = Field(
        description="Pratiques culturales recommandées"
    )
    surveillance: str = Field(description="Fréquence et modalités de surveillance")

    # Synthèse
    resume_diagnostic: str = Field(description="Résumé clair du diagnostic pour l'agriculteur")
    alerte: bool = Field(description="True si une intervention urgente est nécessaire")
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Niveau de confiance global du diagnostic"
    )


class EtatMeteo(BaseModel):
    """Données météo structurées pour l'analyse."""
    temperature_min: float
    temperature_max: float
    humidite: float
    precipitations_mm: float
    vent_kmh: float
    risque_mildiou: str = Field(description="Faible / Modéré / Élevé")
    risque_oïdium: str
    risque_botrytis: str
    source: str = Field(default="OpenMeteo")
