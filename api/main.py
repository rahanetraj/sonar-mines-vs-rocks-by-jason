"""
API FastAPI pour la classification Sonar (Mines vs Rochers).
Endpoints : /health, /info, /predict, /monitoring/stats, /monitoring/recent, /monitoring/drift
"""

import csv
import json
import os
import sys
import subprocess
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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

# Lock thread-safe pour l'écriture CSV et le compteur
_log_lock = threading.Lock()
_counter_lock = threading.Lock()

# État du monitoring
prediction_counter = 0
is_drift_running = False


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


def _run_drift_report_background():
    """Lance le script Evidently en arrière-plan."""
    global is_drift_running
    try:
        print("[MONITORING] Lancement du rapport de drift automatique...")
        subprocess.run(
            [sys.executable, str(ROOT / "monitoring" / "drift_report.py")],
            capture_output=True,
            timeout=120
        )
        print("[MONITORING] Rapport de drift terminé avec succès.")
    except Exception as e:
        print(f"[MONITORING] Erreur lors du rapport de drift : {e}")
    finally:
        is_drift_running = False


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
    model_config = {"protected_namespaces": ()}


class InfoResponse(BaseModel):
    model_type: str
    model_file: str
    scaler_file: str
    features_count: int
    classes: list[str]
    mlflow_experiment: str
    model_config = {"protected_namespaces": ()}


# --- État global de l'application ---
app_state: dict[str, Any] = {}


# --- Lifespan : chargement du modèle au démarrage ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle et le scaler une seule fois au démarrage de l'API."""
    _init_log_file()
    
    # Initialiser le compteur de prédictions à partir du CSV existant
    global prediction_counter
    try:
        with _log_lock:
            if PREDICTIONS_LOG.exists():
                with open(PREDICTIONS_LOG, newline="") as f:
                    # On soustrait 1 pour l'en-tête
                    prediction_counter = sum(1 for _ in f) - 1
                    if prediction_counter < 0: prediction_counter = 0
        print(f"[API] Compteur de prédictions initialisé : {prediction_counter}")
    except Exception as e:
        print(f"[API] Erreur initialisation compteur : {e}")
        prediction_counter = 0

    try:
        predictor = SonarPredictor()
        app_state["predictor"] = predictor
        app_state["model_loaded"] = True
        print("[API] Modèle chargé avec succès.")
    except FileNotFoundError as e:
        app_state["model_loaded"] = False
        print(f"[API] ERREUR : modèle introuvable — {e}")
        print("[API] Lancez d'abord : python src/train.py")
    
    # S'assurer que le dossier des rapports existe pour StaticFiles
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

# --- Servir les rapports HTML statiques ---
REPORTS_DIR = ROOT / "monitoring" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")


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
        
        # Trigger drift report toutes les 10 prédictions
        global prediction_counter, is_drift_running
        with _counter_lock:
            prediction_counter += 1
            trigger_drift = (prediction_counter % 10 == 0)
        
        if trigger_drift and not is_drift_running:
            is_drift_running = True
            Thread(target=_run_drift_report_background, daemon=True).start()
            print(f"[MONITORING] Trigger auto-drift activé (prédiction #{prediction_counter})")
            
    except Exception as e:
        print(f"[API] Erreur logging/monitoring : {e}")

    return PredictResponse(**result)


# ─────────────────────────────────────────
# MONITORING ENDPOINTS
# ─────────────────────────────────────────

@app.get("/monitoring/stats", tags=["Monitoring"])
def monitoring_stats(response: Response):
    """
    Retourne des statistiques agrégées sur toutes les prédictions loggées.
    """
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    global prediction_counter
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
                "next_drift_report_at": 10
            }

        rows: list[dict] = []
        with _log_lock:
            with open(PREDICTIONS_LOG, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

        total      = len(rows)
        # S'assurer que le compteur global est synchronisé
        with _counter_lock:
            prediction_counter = total
        mine_count = sum(1 for r in rows if r.get("prediction") == "Mine")
        rock_count = total - mine_count
        avg_conf   = (
            sum(float(r["confidence"]) for r in rows) / total if total else 0.0
        )
        last_time  = rows[-1]["timestamp"] if rows else None

        # Calculer le nombre de prédictions restantes avant le prochain rapport
        next_at = ((total // 10) + 1) * 10
        remaining = next_at - total

        return {
            "total_predictions"   : total,
            "mine_count"          : mine_count,
            "rock_count"          : rock_count,
            "mine_percentage"     : round(mine_count / total * 100, 2) if total else 0.0,
            "rock_percentage"     : round(rock_count / total * 100, 2) if total else 0.0,
            "avg_confidence"      : round(avg_conf, 4),
            "last_prediction_time": last_time,
            "next_drift_report_at": remaining
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur stats : {str(e)}")


@app.get("/monitoring/recent", tags=["Monitoring"])
def monitoring_recent(response: Response):
    """
    Retourne les 20 dernières prédictions sous forme de tableau JSON.
    """
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
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
def monitoring_drift(response: Response):
    """
    Retourne les métriques de drift depuis le dernier rapport Evidently.
    Générez d'abord le rapport : python monitoring/drift_report.py
    """
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    try:
        if not DRIFT_METRICS.exists():
            return {
                "message": "No drift report yet. Run monitoring/drift_report.py first"
            }
        with open(DRIFT_METRICS) as f:
            data = json.load(f)
            data["auto_generated"] = True
            return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur drift : {str(e)}")
