"""
API FastAPI pour la classification Sonar (Mines vs Rochers).
Endpoints : /health, /info, /predict, /monitoring/stats, /monitoring/recent, /monitoring/drift
"""

import csv
import json
import os
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Ajout du répertoire racine au PYTHONPATH pour importer src.predict
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.predict import SonarPredictor

# Port injecté par Render (ou 8000 par défaut en local)
port = int(os.environ.get("PORT", 8000))

# --- Fichier de log des prédictions ---
PREDICTIONS_LOG = ROOT / "data" / "predictions_log.csv"
DRIFT_METRICS   = ROOT / "monitoring" / "reports" / "drift_metrics.json"

FEATURE_NAMES = [f"f{i}" for i in range(1, 61)]
LOG_COLUMNS   = ["timestamp"] + FEATURE_NAMES + ["prediction", "confidence"]

# Lock thread-safe pour l'écriture CSV
_log_lock = threading.Lock()


def _init_log_file() -> None:
    """Crée le fichier CSV avec les en-têtes s'il n'existe pas."""
    PREDICTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not PREDICTIONS_LOG.exists():
        with open(PREDICTIONS_LOG, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_COLUMNS).writeheader()


def _append_prediction(features: list[float], prediction: str, confidence: float) -> None:
    """Ajoute une ligne de prédiction dans le CSV de manière thread-safe."""
    row = {
        "timestamp"  : datetime.utcnow().isoformat(),
        "prediction" : prediction,
        "confidence" : round(confidence, 6),
    }
    for i, v in enumerate(features, start=1):
        row[f"f{i}"] = v

    with _log_lock:
        with open(PREDICTIONS_LOG, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_COLUMNS).writerow(row)


# --- Modèles Pydantic ---

class PredictRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=60,
        max_length=60,
        description="Exactement 60 valeurs float représentant les bandes de fréquence sonar (entre 0.0 et 1.0)",
        examples=[[0.02, 0.037] + [0.1] * 58],
    )

    @field_validator("features")
    @classmethod
    def validate_feature_range(cls, v: list[float]) -> list[float]:
        """Vérifie que toutes les valeurs sont entre 0.0 et 1.0."""
        for i, val in enumerate(v):
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"La feature à l'index {i} ({val}) doit être entre 0.0 et 1.0"
                )
        return v


class PredictResponse(BaseModel):
    prediction: str       # "Mine" ou "Rock"
    confidence: float     # probabilité de la classe prédite
    probabilities: dict[str, float]  # {"Mine": float, "Rock": float}


class HealthResponse(BaseModel):
    status: str
    model: str


class InfoResponse(BaseModel):
    model_type: str
    model_file: str
    scaler_file: str
    features_count: int
    classes: list[str]
    mlflow_experiment: str


# --- État global de l'application ---
app_state: dict[str, Any] = {}


# --- Lifespan : chargement du modèle au démarrage ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle et le scaler une seule fois au démarrage de l'API."""
    _init_log_file()
    try:
        predictor = SonarPredictor()
        app_state["predictor"] = predictor
        app_state["model_loaded"] = True
        print("[API] Modèle chargé avec succès.")
    except FileNotFoundError as e:
        app_state["model_loaded"] = False
        print(f"[API] ERREUR : modèle introuvable — {e}")
        print("[API] Lancez d'abord : python src/train.py")
    yield
    # Nettoyage (optionnel)
    app_state.clear()


# --- Création de l'application ---
app = FastAPI(
    title="Sonar Classification API",
    description="Classifie des signaux sonar en Mine (M) ou Roche (R) via un RandomForest.",
    version="2.0.0",
    lifespan=lifespan,
)

# --- CORS : autorise les requêtes depuis n'importe quelle origine ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    """Vérifie que l'API et le modèle sont opérationnels."""
    if not app_state.get("model_loaded", False):
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé. Exécutez d'abord python src/train.py.",
        )
    return HealthResponse(status="ok", model="loaded")


@app.get("/info", response_model=InfoResponse, tags=["Monitoring"])
def info():
    """Retourne les métadonnées du modèle chargé."""
    if not app_state.get("model_loaded", False):
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    predictor = app_state["predictor"]
    model_type = type(predictor.model).__name__

    return InfoResponse(
        model_type=model_type,
        model_file="models/model.pkl",
        scaler_file="models/scaler.pkl",
        features_count=60,
        classes=["Rock", "Mine"],
        mlflow_experiment="sonar-classification",
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Classifie un signal sonar.

    - **features** : liste de exactement 60 floats entre 0.0 et 1.0
    - Retourne : classe prédite, score de confiance, probabilités
    """
    if not app_state.get("model_loaded", False):
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé. Exécutez d'abord python src/train.py.",
        )

    try:
        result = app_state["predictor"].predict(request.features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")

    # Log asynchrone (thread-safe, n'impacte pas le temps de réponse)
    try:
        _append_prediction(request.features, result["prediction"], result["confidence"])
    except Exception:
        pass  # Ne jamais bloquer la prédiction à cause du logging

    return PredictResponse(**result)


# ─────────────────────────────────────────
# MONITORING ENDPOINTS
# ─────────────────────────────────────────

@app.get("/monitoring/stats", tags=["Monitoring"])
def monitoring_stats():
    """
    Retourne des statistiques agrégées sur toutes les prédictions loggées.
    """
    try:
        if not PREDICTIONS_LOG.exists():
            return {
                "total_predictions"   : 0,
                "mine_count"          : 0,
                "rock_count"          : 0,
                "mine_percentage"     : 0.0,
                "rock_percentage"     : 0.0,
                "avg_confidence"      : 0.0,
                "last_prediction_time": None,
            }

        rows: list[dict] = []
        with _log_lock:
            with open(PREDICTIONS_LOG, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

        total      = len(rows)
        mine_count = sum(1 for r in rows if r.get("prediction") == "Mine")
        rock_count = total - mine_count
        avg_conf   = (
            sum(float(r["confidence"]) for r in rows) / total if total else 0.0
        )
        last_time  = rows[-1]["timestamp"] if rows else None

        return {
            "total_predictions"   : total,
            "mine_count"          : mine_count,
            "rock_count"          : rock_count,
            "mine_percentage"     : round(mine_count / total * 100, 2) if total else 0.0,
            "rock_percentage"     : round(rock_count / total * 100, 2) if total else 0.0,
            "avg_confidence"      : round(avg_conf, 4),
            "last_prediction_time": last_time,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur stats : {str(e)}")


@app.get("/monitoring/recent", tags=["Monitoring"])
def monitoring_recent():
    """
    Retourne les 20 dernières prédictions sous forme de tableau JSON.
    """
    try:
        if not PREDICTIONS_LOG.exists():
            return []

        with _log_lock:
            with open(PREDICTIONS_LOG, newline="") as f:
                rows = list(csv.DictReader(f))

        # Convertir les valeurs numériques
        recent = []
        for r in rows[-20:]:
            entry = {
                "timestamp"  : r.get("timestamp"),
                "prediction" : r.get("prediction"),
                "confidence" : float(r.get("confidence", 0)),
            }
            recent.append(entry)

        return list(reversed(recent))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur recent : {str(e)}")


@app.get("/monitoring/drift", tags=["Monitoring"])
def monitoring_drift():
    """
    Retourne les métriques de drift depuis le dernier rapport Evidently.
    Générez d'abord le rapport : python monitoring/drift_report.py
    """
    try:
        if not DRIFT_METRICS.exists():
            return {
                "message": "No drift report yet. Run monitoring/drift_report.py first"
            }
        with open(DRIFT_METRICS) as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur drift : {str(e)}")
