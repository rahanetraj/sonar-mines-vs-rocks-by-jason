FROM python:3.11-slim

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Installation des dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie des sources et des données
COPY src/        ./src/
COPY api/        ./api/
COPY data/       ./data/
COPY params.yaml .

# Entraînement du modèle pendant le build (génère models/)
RUN python src/train.py --stage train

# Exposition du port API
EXPOSE 8000

# Commande de démarrage — PORT injecté par Render (défaut 8000)
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
