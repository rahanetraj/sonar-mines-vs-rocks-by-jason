"""
Détection de data drift avec Evidently.
Compare les données de référence (train) avec les données courantes (prédictions récentes).
Génère des rapports HTML dans monitoring/reports/
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Evidently
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric

REPORTS_DIR = ROOT / "monitoring" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = [f"F{i}" for i in range(1, 61)]
# CSV columns use lowercase f1..f60; map to F1..F60 for Evidently
CSV_FEATURE_NAMES = [f"f{i}" for i in range(1, 61)]


# ─────────────────────────────────────────────────────────────
# 1. Chargement des données de référence
# ─────────────────────────────────────────────────────────────
def load_reference_data(params_path: Path = None) -> pd.DataFrame:
    """Charge les données d'entraînement (features uniquement) comme référence."""
    import yaml
    params_path = params_path or ROOT / "params.yaml"
    with open(params_path) as f:
        params = yaml.safe_load(f)

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    data_path = ROOT / params["data"]["path"]
    col_names = FEATURE_NAMES + ["Label"]
    df = pd.read_csv(data_path, header=None, names=col_names)

    le = LabelEncoder()
    df["Target"] = le.fit_transform(df["Label"])
    X = df[FEATURE_NAMES]
    y = df["Target"]

    X_train, _, _, _ = train_test_split(
        X, y,
        test_size=params["model"]["test_size"],
        random_state=params["model"]["random_state"],
        stratify=y,
    )
    print(f"[DRIFT] Données de référence : {X_train.shape[0]} échantillons, {X_train.shape[1]} features")
    return X_train.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# 2. Chargement des données courantes
# ─────────────────────────────────────────────────────────────
def load_current_data(predictions_log: Path = None) -> pd.DataFrame:
    """
    Charge les données courantes depuis data/predictions_log.csv.
    - Ne garde que les colonnes features (f1..f60).
    - Renomme en F1..F60 pour la cohérence avec les données de référence.
    - Quitte si moins de 10 prédictions disponibles.
    """
    predictions_log = predictions_log or ROOT / "data" / "predictions_log.csv"

    if not predictions_log.exists():
        print(f"[DRIFT] ⚠  Fichier introuvable : {predictions_log}")
        print("[DRIFT] Lancez d'abord l'API et envoyez des requêtes POST /predict.")
        sys.exit(1)

    df = pd.read_csv(predictions_log)

    # Vérification du nombre minimum d'échantillons
    n_rows = len(df)
    if n_rows < 10:
        print(f"[DRIFT] ⚠  Seulement {n_rows} prédiction(s) disponible(s) (minimum : 10).")
        print("[DRIFT] Effectuez davantage de prédictions via POST /predict, puis relancez ce script.")
        sys.exit(1)

    # Sélectionner et renommer les colonnes features
    available_csv = [c for c in CSV_FEATURE_NAMES if c in df.columns]
    if len(available_csv) != 60:
        print(f"[DRIFT] ⚠  Colonnes features manquantes dans le log ({len(available_csv)}/60).")
        sys.exit(1)

    current = df[available_csv].copy()
    # Renommer f1→F1, f2→F2, …
    rename_map = {f"f{i}": f"F{i}" for i in range(1, 61)}
    current.rename(columns=rename_map, inplace=True)

    print(f"[DRIFT] Données courantes : {current.shape[0]} prédictions loggées")
    return current.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# 3. Rapport de drift (DataDriftPreset)
# ─────────────────────────────────────────────────────────────
def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: Path = None,
) -> dict:
    """
    Génère le rapport de drift Evidently et retourne un résumé.
    Sauvegarde le rapport HTML dans monitoring/reports/drift_report.html.
    """
    output_path = output_path or REPORTS_DIR / "drift_report.html"

    # Construire les métriques ColumnDriftMetric pour chaque feature
    col_drift_metrics = [ColumnDriftMetric(column_name=f) for f in FEATURE_NAMES]

    report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric(),
        *col_drift_metrics,
    ])
    report.run(reference_data=reference, current_data=current)
    report.save_html(str(output_path))
    print(f"[DRIFT] Rapport drift HTML sauvegardé : {output_path}")

    # Extraction du résumé depuis le dict interne
    result = report.as_dict()
    dataset_drift_result = {}
    drifted_features: list[str] = []

    for metric in result.get("metrics", []):
        metric_id = metric.get("metric", "")
        res = metric.get("result", {})

        if "DatasetDriftMetric" in metric_id:
            dataset_drift_result = {
                "drift_detected": res.get("dataset_drift", False),
                "drift_share"   : res.get("drift_share", 0.0),
                "n_features"    : res.get("number_of_columns", 60),
                "n_drifted"     : res.get("number_of_drifted_columns", 0),
            }

        if "ColumnDriftMetric" in metric_id:
            if res.get("drift_detected"):
                drifted_features.append(res.get("column_name", "?"))

    return {
        "dataset"         : dataset_drift_result,
        "drifted_features": drifted_features,
        "report_path"     : str(output_path),
    }


# ─────────────────────────────────────────────────────────────
# 4. Rapport qualité (DataQualityPreset)
# ─────────────────────────────────────────────────────────────
def generate_quality_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: Path = None,
) -> None:
    """
    Génère le rapport de qualité des données Evidently.
    Sauvegarde dans monitoring/reports/quality_report.html.
    """
    output_path = output_path or REPORTS_DIR / "quality_report.html"

    report = Report(metrics=[DataQualityPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(str(output_path))
    print(f"[DRIFT] Rapport qualité HTML sauvegardé : {output_path}")


# ─────────────────────────────────────────────────────────────
# 5. Affichage du résumé
# ─────────────────────────────────────────────────────────────
def print_summary(summary: dict) -> None:
    """Affiche un résumé lisible du drift détecté."""
    ds      = summary.get("dataset") or {}
    drifted = summary.get("drifted_features", [])

    print("\n" + "=" * 55)
    print("       RÉSUMÉ — RAPPORT DE DATA DRIFT (Evidently)")
    print("=" * 55)
    print(f"  Drift global détecté      : {ds.get('drift_detected', 'N/A')}")
    print(f"  Score de drift            : {ds.get('drift_share', 0):.4f}")
    print(f"  Features driftées         : {ds.get('n_drifted', '?')} / {ds.get('n_features', 60)}")

    if drifted:
        top5 = drifted[:5]
        print(f"  Liste des features        : {', '.join(top5)}", end="")
        if len(drifted) > 5:
            print(f"  … et {len(drifted) - 5} autres", end="")
        print()

    print(f"\n  Rapport HTML drift   : {summary['report_path']}")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────
# 6. Sauvegarde des métriques JSON
# ─────────────────────────────────────────────────────────────
def save_drift_metrics(summary: dict) -> None:
    """
    Sauvegarde les métriques de drift au format JSON normalisé
    dans monitoring/reports/drift_metrics.json.
    """
    ds      = summary.get("dataset") or {}
    drifted = summary.get("drifted_features", [])

    metrics = {
        "timestamp"            : datetime.utcnow().isoformat(),
        "total_features"       : 60,
        "drifted_features"     : ds.get("n_drifted", len(drifted)),
        "drift_detected"       : ds.get("drift_detected", False),
        "drift_score"          : round(ds.get("drift_share", 0.0), 4),
        "drifted_feature_names": drifted,
    }

    out_path = REPORTS_DIR / "drift_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"[DRIFT] Métriques JSON sauvegardées : {out_path}")


# ─────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[DRIFT] Chargement des données de référence (train)…")
    reference = load_reference_data()

    print("[DRIFT] Chargement des données courantes (prédictions loggées)…")
    current = load_current_data()

    print("[DRIFT] Génération du rapport de drift Evidently…")
    summary = generate_drift_report(reference, current)

    print("[DRIFT] Génération du rapport de qualité Evidently…")
    generate_quality_report(reference, current)

    print_summary(summary)
    save_drift_metrics(summary)
    print("\n[DRIFT] ✅ Terminé.")
