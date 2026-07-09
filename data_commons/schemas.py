"""
Schémas Pydantic unifiés pour le Data Commons viticole VITI-AI.
S'appuie sur / complète models/schemas.py (DiagnosticVigne, Maladie, TraitementRecommande).
"""
from __future__ import annotations
from datetime import datetime, date
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SourceType(str, Enum):
    PUBLIQUE = "publique"       # Open-Meteo, IGN, INAO, satellite, etc.
    INTERNE = "interne"         # Mialtech, capteurs propriétaires, robot
    CALCULEE = "calculee"       # dérivée par un agent (agrégations, prédictions)


class TypeCapteur(str, Enum):
    HUMIDITE_SOL = "humidite_sol"
    TEMPERATURE = "temperature"
    NPK = "npk"
    OPTIQUE = "optique"          # caméra multispectrale / RGB
    METEO_STATION = "meteo_station"


# ---------------------------------------------------------------------------
# Entités de référence
# ---------------------------------------------------------------------------

class ParcelleVigne(BaseModel):
    id_parcelle: str
    nom: str
    cepage: str = Field(..., description="ex: Tannat, Petit Manseng, Courbu")
    appellation: str = "Irouléguy"
    latitude: float
    longitude: float
    surface_ha: float
    annee_plantation: Optional[int] = None


class Capteur(BaseModel):
    id_capteur: str
    type_capteur: TypeCapteur
    id_parcelle: str
    latitude: float
    longitude: float
    actif: bool = True
    source: SourceType = SourceType.INTERNE


# ---------------------------------------------------------------------------
# Données mesurées / observées
# ---------------------------------------------------------------------------

class DonneeCapteur(BaseModel):
    id_capteur: str
    id_parcelle: str
    type_capteur: TypeCapteur
    valeur: float
    unite: str
    horodatage: datetime
    source: SourceType = SourceType.INTERNE


class DonneeMeteo(BaseModel):
    id_parcelle: str
    horodatage: datetime
    temperature_c: float
    humidite_pct: float
    precipitations_mm: float
    vitesse_vent_kmh: Optional[float] = None
    source: SourceType = SourceType.PUBLIQUE


class DonneeMaturation(BaseModel):
    id_parcelle: str
    date_mesure: date
    sucre_brix: Optional[float] = None
    acidite_totale_gl: Optional[float] = None
    ph: Optional[float] = None
    poids_100_baies_g: Optional[float] = None
    source: SourceType = SourceType.INTERNE
    methode: str = "manuelle"  # manuelle | robot | drone


class ObservationMaladie(BaseModel):
    id_parcelle: str
    date_observation: date
    maladie: str = Field(..., description="ex: mildiou, oïdium, black rot")
    severite: int = Field(..., ge=0, le=5, description="0=absent, 5=sévère")
    surface_touchee_pct: Optional[float] = None
    image_ref: Optional[str] = Field(None, description="chemin/URL image robot ou drone")
    source: SourceType = SourceType.INTERNE


class TraitementApplique(BaseModel):
    id_parcelle: str
    date_application: date
    produit: str
    dose: str
    cible: str  # maladie ou stress ciblé
    delai_avant_recolte_j: Optional[int] = None
    source: SourceType = SourceType.INTERNE


# ---------------------------------------------------------------------------
# Enveloppe générique (pour ingestion/logging uniforme)
# ---------------------------------------------------------------------------

class DataCommonsRecord(BaseModel):
    table: str
    payload: dict
    source: SourceType
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
