"""
Router FastAPI pour le Data Commons.
À brancher dans api/main.py avec :

    from data_commons.api_router import router as data_commons_router
    app.include_router(data_commons_router, prefix="/data-commons", tags=["data-commons"])
"""
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from data_commons.storage import DataCommonsStore
from data_commons.ingestion import run_ingestion_complete

router = APIRouter()
store = DataCommonsStore()


class ParcelleIn(BaseModel):
    id_parcelle: str
    nom: str
    cepage: str
    appellation: str = "Irouléguy"
    latitude: float
    longitude: float
    surface_ha: float
    annee_plantation: int | None = None


class IngestionRequest(BaseModel):
    parcelles: list[ParcelleIn]
    capteurs_csv: str | None = None
    maturation_csv: str | None = None


@router.post("/parcelles")
def upsert_parcelle(p: ParcelleIn):
    store.upsert_parcelle(p.model_dump())
    return {"status": "ok", "id_parcelle": p.id_parcelle}


@router.get("/parcelles/{id_parcelle}/contexte")
def get_contexte_parcelle(id_parcelle: str, limite: int = 20):
    """
    Contexte structuré consommable directement par l'agent LangGraph
    (nœud `agent` ou `extract_diagnostic`) comme mémoire de travail.
    """
    contexte = store.get_historique_parcelle(id_parcelle, limite)
    if not any(contexte.values()):
        raise HTTPException(status_code=404, detail="Aucune donnée pour cette parcelle")
    return contexte


@router.post("/ingestion/run")
def trigger_ingestion(req: IngestionRequest):
    resume = run_ingestion_complete(
        store,
        parcelles=[p.model_dump() for p in req.parcelles],
        capteurs_csv=req.capteurs_csv,
        maturation_csv=req.maturation_csv,
    )
    return {"status": "ok", "resume": resume}
