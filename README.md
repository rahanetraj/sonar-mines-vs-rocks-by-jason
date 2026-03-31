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

## Interface Web

Un frontend visuel est disponible dans `frontend/index.html` (fichier unique, aucune dépendance sauf Chart.js CDN).

### Ouvrir en local

```bash
# Option 1 — ouvrir directement dans le navigateur
open frontend/index.html          # macOS
xdg-open frontend/index.html      # Linux

# Option 2 — serveur local
python3 -m http.server 3000 --directory frontend
# puis aller sur http://localhost:3000
```

### Héberger sur GitHub Pages

1. Aller dans **Settings → Pages** du repo GitHub
2. Source : `Deploy from a branch`
3. Branch : `main`, dossier : `/frontend`
4. L'interface sera disponible à :
   `https://<username>.github.io/<repo>/`

### Fonctionnalités
- 60 sliders F1–F60 avec visualisation en temps réel (Chart.js)
- Lignes de référence Mine / Rocher sur le graphique
- Boutons : Exemple Mine, Exemple Rocher, Aléatoire, Coller JSON
- Carte de résultat animée avec jauge de confiance
- Historique des 5 dernières prédictions

---

## Résultats du modèle

| Métrique | Score |
|----------|-------|
| Accuracy | 88.1% |
| F1-score (macro) | 87.8% |
| AUC-ROC | 99.1% |

Modèle : **RandomForestClassifier** (100 arbres, random_state=42)

---

## Monitoring

Le pipeline intègre un système complet de monitoring basé sur [Evidently](https://www.evidentlyai.com/) pour détecter le data drift en production.

### Fonctionnement

```
POST /predict
    │
    ▼
data/predictions_log.csv   ← journal de toutes les prédictions (thread-safe)
    │
    ▼
monitoring/drift_report.py ← compare avec les données d'entraînement
    │
    ▼
monitoring/reports/
  ├── drift_report.html     ← rapport Evidently interactif
  ├── quality_report.html   ← rapport qualité des données
  └── drift_metrics.json    ← métriques JSON (lu par /monitoring/drift)
```

### Lancer le rapport de drift manuellement

```bash
# Prérequis : avoir au moins 10 prédictions dans data/predictions_log.csv
python monitoring/drift_report.py
```

Le script :
1. Charge les données d'entraînement comme référence (F1–F60)
2. Charge les prédictions loggées depuis `data/predictions_log.csv`
3. Génère deux rapports HTML Evidently (drift + qualité)
4. Affiche un résumé dans le terminal
5. Sauvegarde `monitoring/reports/drift_metrics.json`

### Endpoints de monitoring

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/monitoring/stats` | Statistiques agrégées (total, % mine/roche, confiance moyenne) |
| `GET` | `/monitoring/recent` | 20 dernières prédictions en JSON |
| `GET` | `/monitoring/drift` | Métriques drift depuis le dernier rapport Evidently |

#### Exemple — `/monitoring/stats`

```bash
curl http://localhost:8000/monitoring/stats
# {
#   "total_predictions": 42,
#   "mine_count": 18,
#   "rock_count": 24,
#   "mine_percentage": 42.86,
#   "rock_percentage": 57.14,
#   "avg_confidence": 0.8721,
#   "last_prediction_time": "2026-03-31T08:30:00.123456"
# }
```

#### Exemple — `/monitoring/drift`

```bash
curl http://localhost:8000/monitoring/drift
# {
#   "timestamp": "2026-03-31T08:30:00",
#   "total_features": 60,
#   "drifted_features": 3,
#   "drift_detected": false,
#   "drift_score": 0.05,
#   "drifted_feature_names": ["F12", "F34", "F47"]
# }
```

### Lire le rapport de drift

- **`drift_detected: true`** → au moins 50 % des features ont drifté (seuil Evidently par défaut)
- **`drift_score`** → proportion de features driftées (0 = aucune, 1 = toutes)
- **`drifted_feature_names`** → liste des features dont la distribution a changé significativement
- Ouvrir `monitoring/reports/drift_report.html` dans un navigateur pour le rapport interactif complet

### Automatisation GitHub Actions

Le workflow `.github/workflows/monitoring.yml` s'exécute :
- **Automatiquement** toutes les 6 heures (`cron: '0 */6 * * *'`)
- **Manuellement** via *Actions → scheduled-monitoring → Run workflow*

Il entraîne le modèle, génère les rapports et commite automatiquement les fichiers mis à jour dans `monitoring/reports/`.

### Interface de monitoring

Le frontend `index.html` affiche un tableau de bord en temps réel :
- **Carte statistiques** : graphique donut Mine/Rocher, confiance moyenne, dernière prédiction
- **Carte drift** : statut (vert/rouge), score sous forme de barre de progression, top 5 features driftées
- Rafraîchissement automatique toutes les **30 secondes**
