"""
Pipeline d'ingestion du Data Commons : orchestre les connecteurs (publics +
internes) vers DataCommonsStore. Conçu pour tourner en tâche planifiée
(cron / APScheduler) ou être appelé manuellement/par un agent.
"""
from __future__ import annotations
import logging
from typing import Iterable

from data_commons.storage import DataCommonsStore
from data_commons.connectors import meteo_connector, capteurs_connector, mialtech_connector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data_commons.ingestion")


def ingest_meteo_toutes_parcelles(store: DataCommonsStore, parcelles: Iterable[dict]) -> int:
    """parcelles: liste de dicts {id_parcelle, latitude, longitude}."""
    n = 0
    for p in parcelles:
        try:
            data = meteo_connector.fetch_meteo_parcelle(
                p["id_parcelle"], p["latitude"], p["longitude"]
            )
            store.add_donnee_meteo(data)
            n += 1
        except Exception as e:
            logger.warning("Échec ingestion météo parcelle %s: %s", p["id_parcelle"], e)
    logger.info("Ingestion météo: %d parcelles mises à jour", n)
    return n


def ingest_capteurs_csv(store: DataCommonsStore, csv_path: str) -> int:
    n = 0
    for row in capteurs_connector.read_capteurs_csv(csv_path):
        store.add_donnee_capteur(row)
        n += 1
    logger.info("Ingestion capteurs: %d mesures importées", n)
    return n


def ingest_maturation_csv(store: DataCommonsStore, csv_path: str) -> int:
    n = 0
    for row in mialtech_connector.read_maturation_csv(csv_path):
        store.add_donnee_maturation(row)
        n += 1
    logger.info("Ingestion maturation (Mialtech): %d mesures importées", n)
    return n


def run_ingestion_complete(
    store: DataCommonsStore,
    parcelles: list[dict],
    capteurs_csv: str | None = None,
    maturation_csv: str | None = None,
) -> dict:
    """Orchestre une passe d'ingestion complète. Retourne un résumé."""
    resume = {}
    for p in parcelles:
        store.upsert_parcelle(p)
    resume["meteo"] = ingest_meteo_toutes_parcelles(store, parcelles)
    if capteurs_csv:
        resume["capteurs"] = ingest_capteurs_csv(store, capteurs_csv)
    if maturation_csv:
        resume["maturation"] = ingest_maturation_csv(store, maturation_csv)
    return resume


if __name__ == "__main__":
    # Exemple d'exécution manuelle avec la parcelle pilote Irouléguy
    store = DataCommonsStore()
    parcelle_pilote = [
        {
            "id_parcelle": "irouleguy-pilote-01",
            "nom": "Parcelle pilote Fructhor",
            "cepage": "Tannat",
            "appellation": "Irouléguy",
            "latitude": 43.2601,
            "longitude": -1.2245,
            "surface_ha": 1.2,
            "annee_plantation": 2010,
        }
    ]
    print(run_ingestion_complete(store, parcelle_pilote))
