"""
Logique de prédiction réutilisable par l'API et les scripts CLI.
Charge le modèle et le scaler une seule fois au démarrage.
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# Mapping numérique → étiquette lisible
LABEL_MAP = {0: "Rock", 1: "Mine"}

class SonarPredictor:
    """Encapsule le modèle et le scaler pour des prédictions en ligne."""

    def __init__(self, model_path: Path = None, scaler_path: Path = None):
        model_path  = model_path  or MODELS_DIR / "model.pkl"
        scaler_path = scaler_path or MODELS_DIR / "scaler.pkl"

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        print(f"[PREDICT] Modèle chargé : {model_path.name}")
        print(f"[PREDICT] Scaler chargé : {scaler_path.name}")

    def predict(self, features: list[float]) -> dict[str, Any]:
        """
        Prédit la classe d'un échantillon sonar.

        Args:
            features: liste de 60 valeurs float entre 0.0 et 1.0

        Returns:
            dict avec 'prediction', 'confidence', 'probabilities'
        """
        X = np.array(features, dtype=np.float64).reshape(1, -1)
        X_sc = self.scaler.transform(X)

        pred_class = int(self.model.predict(X_sc)[0])
        proba      = self.model.predict_proba(X_sc)[0]   # [P(Rock), P(Mine)]

        return {
            "prediction"   : LABEL_MAP[pred_class],
            "confidence"   : float(proba[pred_class]),
            "probabilities": {
                "Rock": float(proba[0]),
                "Mine": float(proba[1]),
            },
        }

    def predict_batch(self, feature_matrix: list[list[float]]) -> list[dict]:
        """Prédictions sur un lot d'échantillons."""
        return [self.predict(row) for row in feature_matrix]


# Point d'entrée CLI
if __name__ == "__main__":
    import sys
    import json

    # Exemple d'utilisation : python src/predict.py '{"features": [0.02, ...]}'
    if len(sys.argv) < 2:
        # Données de démonstration (premier échantillon du dataset)
        demo = [0.02, 0.0371, 0.0428, 0.0207, 0.0954, 0.0986, 0.1539, 0.1601,
                0.3109, 0.2111, 0.1609, 0.1582, 0.2238, 0.0645, 0.0660, 0.2273,
                0.3100, 0.2999, 0.5078, 0.4797, 0.5783, 0.5071, 0.4328, 0.5550,
                0.6711, 0.6415, 0.7104, 0.8080, 0.6791, 0.3857, 0.1307, 0.2604,
                0.5121, 0.7547, 0.8537, 0.8507, 0.6692, 0.6097, 0.4943, 0.2744,
                0.0510, 0.2834, 0.2825, 0.4256, 0.2641, 0.1386, 0.1051, 0.1343,
                0.0383, 0.0324, 0.0232, 0.0027, 0.0065, 0.0159, 0.0072, 0.0167,
                0.0180, 0.0084, 0.0090, 0.0032]
        payload = {"features": demo}
    else:
        payload = json.loads(sys.argv[1])

    predictor = SonarPredictor()
    result = predictor.predict(payload["features"])
    print(json.dumps(result, indent=2))
