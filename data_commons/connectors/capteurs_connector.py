"""
Connecteur source INTERNE : capteurs IoT (humidité sol, NPK, optique, station météo locale).

V1 : lecture d'un CSV d'export (format Raspberry Pi / Arduino Mega évoqué dans la
fiche de stage). Prévu pour être remplacé par un flux MQTT / RS-485 en V2 sans
changer la signature de `parse_and_ingest`.

Format CSV attendu (une ligne = une mesure) :
id_capteur,id_parcelle,type_capteur,valeur,unite,horodatage
"""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Iterator


def read_capteurs_csv(csv_path: str | Path) -> Iterator[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield {
                "id_capteur": row["id_capteur"],
                "id_parcelle": row["id_parcelle"],
                "type_capteur": row["type_capteur"],
                "valeur": float(row["valeur"]),
                "unite": row["unite"],
                "horodatage": row["horodatage"],
                "source": "interne",
            }


# Point d'extension futur : remplacer par un client MQTT (paho-mqtt) qui
# publie directement vers DataCommonsStore.add_donnee_capteur via ingestion.py
def read_capteurs_mqtt_stub(*args, **kwargs):
    raise NotImplementedError(
        "Connecteur MQTT non encore implémenté. "
        "Prévu pour la phase d'intégration robot/capteurs RS-485."
    )
