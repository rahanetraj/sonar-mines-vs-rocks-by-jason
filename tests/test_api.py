"""
Tests pytest pour l'API FastAPI Sonar.
Utilise TestClient de httpx (via starlette).
"""

import random
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ajout du répertoire racine au PYTHONPATH
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.main import app

# Graine pour reproductibilité des données de test
random.seed(42)

# Données de test valides : 60 floats entre 0.0 et 1.0
VALID_FEATURES = [round(random.uniform(0.0, 1.0), 4) for _ in range(60)]


@pytest.fixture(scope="module")
def client():
    """Client de test partagé pour tous les tests du module."""
    with TestClient(app) as c:
        yield c


# ─────────────────────────────────────────────────────
# TEST 1 : /health
# ─────────────────────────────────────────────────────
def test_health_endpoint(client: TestClient):
    """L'endpoint /health doit retourner status=ok et model=loaded."""
    response = client.get("/health")
    assert response.status_code == 200, f"Attendu 200, obtenu {response.status_code}"
    data = response.json()
    assert data["status"] == "ok"
    assert data["model"] == "loaded"


# ─────────────────────────────────────────────────────
# TEST 2 : /predict avec une entrée valide
# ─────────────────────────────────────────────────────
def test_predict_valid_input(client: TestClient):
    """Une requête valide (60 features ∈ [0,1]) doit retourner 200."""
    payload = {"features": VALID_FEATURES}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Attendu 200, obtenu {response.status_code}"
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "probabilities" in data


# ─────────────────────────────────────────────────────
# TEST 3 : /predict avec 59 features → 422
# ─────────────────────────────────────────────────────
def test_predict_wrong_features_count(client: TestClient):
    """Une requête avec 59 features doit retourner 422 (Unprocessable Entity)."""
    payload = {"features": VALID_FEATURES[:59]}  # 59 features au lieu de 60
    response = client.post("/predict", json=payload)
    assert response.status_code == 422, f"Attendu 422, obtenu {response.status_code}"


# ─────────────────────────────────────────────────────
# TEST 4 : /predict avec valeur hors plage → 422
# ─────────────────────────────────────────────────────
def test_predict_invalid_values(client: TestClient):
    """Une feature > 1.0 doit retourner 422."""
    invalid_features = VALID_FEATURES.copy()
    invalid_features[10] = 1.5   # Valeur hors plage [0.0, 1.0]
    payload = {"features": invalid_features}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422, f"Attendu 422, obtenu {response.status_code}"


# ─────────────────────────────────────────────────────
# TEST 5 : /predict retourne "Mine" ou "Rock"
# ─────────────────────────────────────────────────────
def test_predict_returns_mine_or_rock(client: TestClient):
    """La prédiction doit être exactement 'Mine' ou 'Rock'."""
    payload = {"features": VALID_FEATURES}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    prediction = response.json()["prediction"]
    assert prediction in ("Mine", "Rock"), (
        f"Prédiction invalide : '{prediction}' (attendu 'Mine' ou 'Rock')"
    )


# ─────────────────────────────────────────────────────
# TEST 6 : /info
# ─────────────────────────────────────────────────────
def test_info_endpoint(client: TestClient):
    """L'endpoint /info doit retourner les métadonnées du modèle."""
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert data["features_count"] == 60
    assert set(data["classes"]) == {"Mine", "Rock"}


# ─────────────────────────────────────────────────────
# TEST 7 : confiance entre 0 et 1
# ─────────────────────────────────────────────────────
def test_predict_confidence_range(client: TestClient):
    """Le score de confiance doit être entre 0.0 et 1.0."""
    payload = {"features": VALID_FEATURES}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    confidence = response.json()["confidence"]
    assert 0.0 <= confidence <= 1.0, f"Confiance hors plage : {confidence}"
