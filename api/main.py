"""
API FastAPI pour la classification Sonar (Mines vs Rochers).
Endpoints : /health, /info, /predict
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# Ajout du répertoire racine au PYTHONPATH pour importer src.predict
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.predict import SonarPredictor

# Port injecté par Render (ou 8000 par défaut en local)
port = int(os.environ.get("PORT", 8000))

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
    version="1.0.0",
    lifespan=lifespan,
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

    return PredictResponse(**result)
