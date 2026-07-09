"""
Connecteur source INTERNE : base Mialtech (taille de cep de vigne pour maturation),
citée dans la fiche de stage comme source privée du DATA COMMONS viticole.

V1 : import CSV. À remplacer par un connecteur API/BD directe si Mialtech
expose un accès programmatique.

Format CSV attendu :
id_parcelle,date_mesure,sucre_brix,acidite_totale_gl,ph,poids_100_baies_g,methode
"""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Iterator


def read_maturation_csv(csv_path: str | Path) -> Iterator[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield {
                "id_parcelle": row["id_parcelle"],
                "date_mesure": row["date_mesure"],
                "sucre_brix": float(row["sucre_brix"]) if row.get("sucre_brix") else None,
                "acidite_totale_gl": float(row["acidite_totale_gl"]) if row.get("acidite_totale_gl") else None,
                "ph": float(row["ph"]) if row.get("ph") else None,
                "poids_100_baies_g": float(row["poids_100_baies_g"]) if row.get("poids_100_baies_g") else None,
                "source": "interne",
                "methode": row.get("methode", "manuelle"),
            }
