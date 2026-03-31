# Sonar Classification — Mines vs Rocks (MLOps Pipeline)

Classification de signaux sonar en **Mine** ou **Roche** via un pipeline MLOps complet.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MLOPS PIPELINE                         │
│                                                             │
│   data/CSV ──► DVC ──► src/train.py ──► models/            │
│                              │                              │
│                           MLflow                            │
│                         (./mlruns)                          │
│                              │                              │
│                    src/evaluate.py                          │
│                              │                              │
│                         api/main.py  ◄──  src/predict.py   │
│                         (FastAPI)                           │
│                              │                              │
│                        Dockerfile                           │
│                        docker-compose                       │
│                              │                              │
│              .github/workflows/ci-cd.yml                    │
│              (test → deploy via Render webhook)             │
│                              │                              │
│              monitoring/drift_report.py                     │
│                         (Evidently)                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Démarrage rapide

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Entraînement

```bash
python src/train.py --stage train
```

### 3. Évaluation

```bash
python src/evaluate.py
```

### 4. Démarrage de l'API

```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Tests

```bash
pytest tests/ -v
```

### 6. Docker

```bash
# Build + run (API + MLflow)
docker-compose up --build

# Build seul
docker build -t sonar-api .
```

### 7. Drift monitoring

```bash
python monitoring/drift_report.py
# Rapport HTML → monitoring/reports/drift_report.html
```

---

## API — Exemples d'utilisation

### Health check

```bash
curl http://localhost:8000/health
# {"status":"ok","model":"loaded"}
```

### Info modèle

```bash
curl http://localhost:8000/info
```

### Prédiction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.02, 0.037, 0.043, 0.021, 0.095, 0.099, 0.154, 0.160,
                    0.311, 0.211, 0.161, 0.158, 0.224, 0.065, 0.066, 0.227,
                    0.310, 0.300, 0.508, 0.480, 0.578, 0.507, 0.433, 0.555,
                    0.671, 0.642, 0.710, 0.808, 0.679, 0.386, 0.131, 0.260,
                    0.512, 0.755, 0.854, 0.851, 0.669, 0.610, 0.494, 0.274,
                    0.051, 0.283, 0.283, 0.426, 0.264, 0.139, 0.105, 0.134,
                    0.038, 0.032, 0.023, 0.003, 0.007, 0.016, 0.007, 0.017,
                    0.018, 0.008, 0.009, 0.003]}'

# {"prediction":"Rock","confidence":0.87,"probabilities":{"Mine":0.13,"Rock":0.87}}
```

---

## Pipeline DVC

```bash
dvc repro        # rejouer toutes les étapes
dvc dag          # visualiser le graphe
dvc metrics show # afficher les métriques
```

---

## Déploiement sur Render.com

### Étapes
1. Pousser le repo sur GitHub
2. Aller sur [render.com](https://render.com) → **New Web Service**
3. Connecter le repo GitHub
4. Render détecte automatiquement le `render.yaml` (Blueprint)
5. Ajouter le secret `RENDER_DEPLOY_HOOK` dans GitHub → Settings → Secrets → Actions

### URL publique
```
https://sonar-api.onrender.com
```

### Exemples d'utilisation en production

```bash
# Health check
curl https://sonar-api.onrender.com/health

# Prédiction
curl -X POST https://sonar-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.02, 0.037, 0.043, 0.021, 0.095, 0.099, 0.154, 0.160,
                    0.311, 0.211, 0.161, 0.158, 0.224, 0.065, 0.066, 0.227,
                    0.310, 0.300, 0.508, 0.480, 0.578, 0.507, 0.433, 0.555,
                    0.671, 0.642, 0.710, 0.808, 0.679, 0.386, 0.131, 0.260,
                    0.512, 0.755, 0.854, 0.851, 0.669, 0.610, 0.494, 0.274,
                    0.051, 0.283, 0.283, 0.426, 0.264, 0.139, 0.105, 0.134,
                    0.038, 0.032, 0.023, 0.003, 0.007, 0.016, 0.007, 0.017,
                    0.018, 0.008, 0.009, 0.003]}'
```

### Secret GitHub requis

| Secret | Description |
|--------|-------------|
| `RENDER_DEPLOY_HOOK` | URL webhook fournie par Render (Settings → Deploy Hook) |

---

## Résultats du modèle

| Métrique | Score |
|----------|-------|
| Accuracy | 88.1% |
| F1-score (macro) | 87.8% |
| AUC-ROC | 99.1% |

Modèle : **RandomForestClassifier** (100 arbres, random_state=42)
