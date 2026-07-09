"""
Connecteur source PUBLIQUE : Open-Meteo.
Réutilise/adapte la logique de tools/vigne_tools.py::get_meteo_vigne,
mais ici pour l'ingestion en masse dans le Data Commons (pas pour un appel
ponctuel de l'agent).
"""
from __future__ import annotations
import requests
from datetime import datetime, timezone

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_meteo_parcelle(id_parcelle: str, latitude: float, longitude: float) -> dict:
    """Récupère les conditions météo courantes pour une parcelle donnée."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "timezone": "Europe/Paris",
    }
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()["current"]

    return {
        "id_parcelle": id_parcelle,
        "horodatage": datetime.now(timezone.utc).isoformat(),
        "temperature_c": data.get("temperature_2m"),
        "humidite_pct": data.get("relative_humidity_2m"),
        "precipitations_mm": data.get("precipitation", 0.0),
        "vitesse_vent_kmh": data.get("wind_speed_10m"),
        "source": "publique",
    }
