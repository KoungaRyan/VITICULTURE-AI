"""
Couche de stockage du Data Commons viticole.

Choix : SQLite pour démarrer (cohérent avec le SqliteSaver déjà utilisé pour
les checkpoints LangGraph), avec une interface pensée pour migrer plus tard
vers PostgreSQL sans changer les appelants (mêmes méthodes, autre backend).
"""
from __future__ import annotations
import sqlite3
import json
import os
from contextlib import contextmanager
from typing import Iterator

DB_PATH = os.environ.get("DATA_COMMONS_DB_PATH", "data/data_commons.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS parcelles (
    id_parcelle TEXT PRIMARY KEY,
    nom TEXT, cepage TEXT, appellation TEXT,
    latitude REAL, longitude REAL, surface_ha REAL, annee_plantation INTEGER
);

CREATE TABLE IF NOT EXISTS capteurs (
    id_capteur TEXT PRIMARY KEY,
    type_capteur TEXT, id_parcelle TEXT,
    latitude REAL, longitude REAL, actif INTEGER, source TEXT,
    FOREIGN KEY (id_parcelle) REFERENCES parcelles(id_parcelle)
);

CREATE TABLE IF NOT EXISTS donnees_capteurs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    id_capteur TEXT, id_parcelle TEXT, type_capteur TEXT,
    valeur REAL, unite TEXT, horodatage TEXT, source TEXT
);

CREATE TABLE IF NOT EXISTS donnees_meteo (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    id_parcelle TEXT, horodatage TEXT,
    temperature_c REAL, humidite_pct REAL, precipitations_mm REAL,
    vitesse_vent_kmh REAL, source TEXT
);

CREATE TABLE IF NOT EXISTS donnees_maturation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    id_parcelle TEXT, date_mesure TEXT,
    sucre_brix REAL, acidite_totale_gl REAL, ph REAL,
    poids_100_baies_g REAL, source TEXT, methode TEXT
);

CREATE TABLE IF NOT EXISTS observations_maladies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    id_parcelle TEXT, date_observation TEXT, maladie TEXT,
    severite INTEGER, surface_touchee_pct REAL, image_ref TEXT, source TEXT
);

CREATE TABLE IF NOT EXISTS traitements_appliques (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    id_parcelle TEXT, date_application TEXT, produit TEXT, dose TEXT,
    cible TEXT, delai_avant_recolte_j INTEGER, source TEXT
);

CREATE TABLE IF NOT EXISTS ingestion_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_cible TEXT, source TEXT, ingested_at TEXT, payload_json TEXT
);
"""


class DataCommonsStore:
    """Point d'accès unique en lecture/écriture au Data Commons."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    # -- écriture générique avec log d'ingestion --------------------------
    def _insert(self, table: str, row: dict, source: str) -> None:
        cols = ", ".join(row.keys())
        placeholders = ", ".join(["?"] * len(row))
        with self._conn() as conn:
            conn.execute(
                f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
                list(row.values()),
            )
            conn.execute(
                "INSERT INTO ingestion_log (table_cible, source, ingested_at, payload_json) "
                "VALUES (?, ?, datetime('now'), ?)",
                (table, source, json.dumps(row, default=str)),
            )

    # -- API par entité (utilisée par ingestion.py et les tools agents) ---
    def upsert_parcelle(self, p: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO parcelles (id_parcelle, nom, cepage, appellation,
                       latitude, longitude, surface_ha, annee_plantation)
                   VALUES (:id_parcelle, :nom, :cepage, :appellation,
                       :latitude, :longitude, :surface_ha, :annee_plantation)
                   ON CONFLICT(id_parcelle) DO UPDATE SET
                       nom=excluded.nom, cepage=excluded.cepage,
                       appellation=excluded.appellation, latitude=excluded.latitude,
                       longitude=excluded.longitude, surface_ha=excluded.surface_ha,
                       annee_plantation=excluded.annee_plantation""",
                p,
            )

    def add_donnee_capteur(self, d: dict) -> None:
        self._insert("donnees_capteurs", d, d.get("source", "interne"))

    def add_donnee_meteo(self, d: dict) -> None:
        self._insert("donnees_meteo", d, d.get("source", "publique"))

    def add_donnee_maturation(self, d: dict) -> None:
        self._insert("donnees_maturation", d, d.get("source", "interne"))

    def add_observation_maladie(self, d: dict) -> None:
        self._insert("observations_maladies", d, d.get("source", "interne"))

    def add_traitement(self, d: dict) -> None:
        self._insert("traitements_appliques", d, d.get("source", "interne"))

    # -- lecture pour les agents (RAG structuré / contexte) ---------------
    def get_historique_parcelle(self, id_parcelle: str, limite: int = 20) -> dict:
        """Vue agrégée utilisée par l'agent LangGraph comme contexte de décision."""
        with self._conn() as conn:
            meteo = conn.execute(
                "SELECT * FROM donnees_meteo WHERE id_parcelle=? "
                "ORDER BY horodatage DESC LIMIT ?", (id_parcelle, limite),
            ).fetchall()
            maturation = conn.execute(
                "SELECT * FROM donnees_maturation WHERE id_parcelle=? "
                "ORDER BY date_mesure DESC LIMIT ?", (id_parcelle, limite),
            ).fetchall()
            maladies = conn.execute(
                "SELECT * FROM observations_maladies WHERE id_parcelle=? "
                "ORDER BY date_observation DESC LIMIT ?", (id_parcelle, limite),
            ).fetchall()
            traitements = conn.execute(
                "SELECT * FROM traitements_appliques WHERE id_parcelle=? "
                "ORDER BY date_application DESC LIMIT ?", (id_parcelle, limite),
            ).fetchall()
        return {
            "meteo": [dict(r) for r in meteo],
            "maturation": [dict(r) for r in maturation],
            "maladies": [dict(r) for r in maladies],
            "traitements": [dict(r) for r in traitements],
        }
