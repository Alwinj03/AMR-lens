"""
AMR Lens prediction API (runs as a Vercel Python Function).

POST /api/predict
  {"iso3": "IND", "year": 2020, "specimen": "Blood",
   "pathogen": "Escherichia coli", "antibiotic": "Ceftriaxone"}
-> estimates from the additive baseline and the random forest.

Only combinations of pathogen, antibiotic and specimen that exist somewhere
in the GLASS data are accepted: anything else would be extrapolation the
model has no basis for.
"""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel

BUNDLE = joblib.load(Path(__file__).parent / "model.joblib")  # once per cold start
META = BUNDLE["meta"]
COUNTRY = {c["iso3"]: c for c in META["countries"]}
COMBOS = set(META["combos"])
YEARS = META["years"]

app = FastAPI(title="AMR Lens API")
router = APIRouter()


class Query(BaseModel):
    iso3: str
    year: int
    specimen: str
    pathogen: str
    antibiotic: str


@router.get("/health")
def health():
    return {"ok": True, "countries": len(COUNTRY), "combos": len(COMBOS)}


@router.post("/predict")
def predict(q: Query):
    iso3 = q.iso3.upper()
    if iso3 not in COUNTRY:
        raise HTTPException(422, f"Unknown country code '{q.iso3}'.")
    if q.year not in YEARS:
        raise HTTPException(422, f"Year must be one of {YEARS}; the model does not forecast.")
    combo = f"{q.pathogen} | {q.antibiotic} | {q.specimen}"
    if combo not in COMBOS:
        raise HTTPException(422, "This pathogen, antibiotic and specimen combination "
                                 "is not reported anywhere in GLASS.")
    row = pd.DataFrame([{
        "iso3": iso3, "region": COUNTRY[iso3]["region"], "year": q.year,
        "specimen": q.specimen, "pathogen": q.pathogen,
        "antibiotic": q.antibiotic, "combo": combo,
    }])[BUNDLE["features"]]
    out = {}
    for name in ("additive", "forest"):
        out[name] = round(float(min(100, max(0, BUNDLE[name].predict(row)[0]))), 1)
    return {"estimates": out, "country": COUNTRY[iso3]["country"], "combo": combo}


# Mounted at both paths so routing works whether or not Vercel strips the /api prefix.
app.include_router(router, prefix="/api")
app.include_router(router)
