#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fase 5 - Evaluación integral (CRISP-DM)
- Métricas: Accuracy, Precision, Recall, F1, ROC-AUC
- Matriz de confusión
- Curvas ROC y Precision-Recall
- Importancia de variables: Permutation Importance (y SHAP opcional)
- Validación cruzada temporal (Time-based CV por JoinDate)
- Análisis de errores (FP/FN de alta confianza)
- Impacto de negocio esperado (parámetros por CLI)

Artefactos guardados y logueados en MLflow.
"""

import os, json, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from pathlib import Path
from datetime import datetime
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             RocCurveDisplay, PrecisionRecallDisplay)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

# ========= util =========

def ensure_dir(p): Path(p).parent.mkdir(parents=True, exist_ok=True)

def load_splits(proc_dir: str):
    with open(os.path.join(proc_dir,"columns.json")) as f:
        meta = json.load(f)
    train = pd.read_csv(os.path.join(proc_dir, "train.csv"))
    val   = pd.read_csv(os.path.join(proc_dir, "val.csv"))
    test  = pd.read_csv(os.path.join(proc_dir, "test.csv"))
    target = meta["target"]; drops = meta["drop_cols"]
    num_cols = meta["features_num"]; cat_cols = meta["features_cat"]

    def split_xy(df):
        X = df.drop(columns=[target]+drops, errors="ignore")
        y = df[target].astype(int).values
        return X, y, df  # devuelve además el df por JoinDate

    Xtr, ytr, dtr = split_xy(train)
    Xva, yva, dva = split_xy(val)
    Xte, yte, dte = split_xy(test)
    return (Xtr,ytr,dtr),(Xva,yva,dva),(Xte,yte,dte), num_cols, cat_cols, target

def find_best_run_id(experiment_name: str, client: MlflowClient):
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"Experiment '{experiment_name}' not found in MLflow.")
    runs = client.search_runs([exp.experiment_id], order_by=["metrics.val_f1 DESC"], max_results=1)
    if not runs:
        raise RuntimeError(f"No runs found in experiment '{experiment_name}'.")
    return runs[0].info.run_id

def load_model_from_run(run_id: str, model_stage: str = "final_model"):
    uri = f"runs:/{run_id}/{model_stage}"
    return mlflow.sklearn.load_model(uri)

def eval_and_log_basic(y_true, y_prob, out_dir: str, prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }
    for k,v in metrics.items():
        mlflow.log_metric(f"{prefix}_{k}", float(v))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{prefix} - Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for (i,j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha="center", va="center")
    fig.colorbar(im); fig.tight_layout()
    p_cm = os.path.join(out_dir, f"{prefix}_confusion_matrix.png")
    plt.savefig(p_cm, dpi=140); plt.close()
    mlflow.log_artifact(p_cm)

    # ROC
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title(f"{prefix} - ROC Curve")
    fig.tight_layout()
    p_roc = os.path.join(out_dir, f"{prefix}_roc.png")
    plt.savefig(p_roc, dpi=140); plt.close()
    mlflow.log_artifact(p_roc)

    # PR
    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title(f"{prefix} - Precision-Recall Curve")
    fig.tight_layout()
    p_pr = os.path.join(out_dir, f"{prefix}_pr.png")
    plt.savefig(p_pr, dpi=140); plt.close()
    mlflow.log_artifact(p_pr)

    # save metrics json
    with open(os.path.join(out_dir, f"{prefix}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    mlflow.log_artifact(os.path.join(out_dir, f"{prefix}_metrics.json"))

    return metrics, cm

def temporal_cv(model, X_full, y_full, dates, n_splits: int, out_dir: str):
    """Time-based CV usando TimeSeriesSplit respetando el orden por dates (JoinDate)."""
    os.makedirs(out_dir, exist_ok=True)
    # ordenar por fecha
    order = np.argsort(pd.to_datetime(dates).values)
    X_ord = X_full.iloc[order]
    y_ord = y_full[order]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for i, (tr_idx, te_idx) in enumerate(tscv.split(X_ord)):
        Xtr, Xte = X_ord.iloc[tr_idx], X_ord.iloc[te_idx]
        ytr, yte = y_ord[tr_idx], y_ord[te_idx]
        # re-ajusta el pipeline entero (prep+clf)
        model.fit(Xtr, ytr)
        y_prob = model.predict_proba(Xte)[:,1]
        f1 = f1_score(yte, (y_prob>=0.5).astype(int), zero_division=0)
        auc = roc_auc_score(yte, y_prob)
        scores.append({"fold": i+1, "f1": f1, "roc_auc": auc})

    df_scores = pd.DataFrame(scores)
    df_scores.to_csv(os.path.join(out_dir, "temporal_cv_scores.csv"), index=False)
    mlflow.log_artifact(os.path.join(out_dir, "temporal_cv_scores.csv"))

    fig, ax = plt.subplots()
    df_scores[["f1","roc_auc"]].plot(kind="bar", ax=ax)
    ax.set_title("Temporal CV scores (F1 / ROC-AUC)")
    ax.set_xticklabels([f"Fold {r}" for r in df_scores["fold"]], rotation=0)
    fig.tight_layout()
    p = os.path.join(out_dir, "temporal_cv_bars.png")
    plt.savefig(p, dpi=140); plt.close()
    mlflow.log_artifact(p)

    mlflow.log_metric("temporal_cv_f1_mean", float(df_scores["f1"].mean()))
    mlflow.log_metric("temporal_cv_roc_auc_mean", float(df_scores["roc_auc"].mean()))
    return df_scores

def permutation_importance_report(model, X, y, feature_names, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    r = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1, scoring="f1")
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std
    }).sort_values("importance_mean", ascending=False)
    df_imp.to_csv(os.path.join(out_dir, "permutation_importance.csv"), index=False)
    mlflow.log_artifact(os.path.join(out_dir, "permutation_importance.csv"))

    # plot top 20
    top = df_imp.head(20).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(6,7))
    ax.barh(top["feature"], top["importance_mean"])
    ax.set_title("Permutation Importance (top 20)")
    fig.tight_layout()
    p = os.path.join(out_dir, "permutation_importance_top20.png")
    plt.savefig(p, dpi=140); plt.close()
    mlflow.log_artifact(p)
    return df_imp

def shap_report(model, X_sample, out_dir: str):
    """Reporte de shap para explicar decisiones del modelo"""
    import shap
    os.makedirs(out_dir, exist_ok=True)
    explainer = shap.Explainer(model.named_steps["clf"])  # intenta explicar el último paso
    # Nota: para pipelines con OHE/Scaler, lo ideal es explicar el pipeline transformado;
    # aquí usamos un muestreo de 100 filas transformadas:
    X_trans = model.named_steps["prep"].transform(X_sample)
    shap_values = explainer(X_trans)
    p = os.path.join(out_dir, "shap_summary.png")
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout(); plt.savefig(p, dpi=140); plt.close()
    mlflow.log_artifact(p)
    return True


def error_analysis(y_true, y_prob, X_df, out_dir: str, topk: int = 50):
    os.makedirs(out_dir, exist_ok=True)
    y_pred = (y_prob >= 0.5).astype(int)
    df_err = pd.DataFrame({
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred
    }).reset_index(drop=True)
    # falsos positivos y falsos negativos con mayor confianza
    fp = df_err[(df_err.y_true==0) & (df_err.y_pred==1)].sort_values("y_prob", ascending=False).head(topk)
    fn = df_err[(df_err.y_true==1) & (df_err.y_pred==0)].sort_values("y_prob", ascending=True).head(topk)

    fp.to_csv(os.path.join(out_dir, "top_false_positives.csv"), index=False)
    fn.to_csv(os.path.join(out_dir, "top_false_negatives.csv"), index=False)
    mlflow.log_artifact(os.path.join(out_dir, "top_false_positives.csv"))
    mlflow.log_artifact(os.path.join(out_dir, "top_false_negatives.csv"))
    return fp, fn

def business_impact(y_true, y_prob, avg_monthly_revenue: float,
                    intervention_cost: float, intervention_rate: float,
                    retention_success_rate: float, out_dir: str):
    """
    Estima impacto suponiendo que intervienes al top 'intervention_rate' de probabilidades.
    - Si era churn (1) y se interviene, hay prob de retener = retention_success_rate
    - Beneficio = clientes retenidos * avg_monthly_revenue (1 mes) - costo intervenciones
    """
    os.makedirs(out_dir, exist_ok=True)
    n = len(y_true)
    k = max(1, int(n * intervention_rate))
    idx_top = np.argsort(y_prob)[::-1][:k]

    # métricas de targeting
    y_top = y_true[idx_top]
    num_true_churn_targeted = int(y_top.sum())
    retained = int(num_true_churn_targeted * retention_success_rate)

    benefit = retained * avg_monthly_revenue
    cost = k * intervention_cost
    net = benefit - cost

    report = {
        "n_samples": n,
        "k_targeted": k,
        "top_rate": intervention_rate,
        "true_churn_in_topk": num_true_churn_targeted,
        "retention_success_rate": retention_success_rate,
        "retained_customers": retained,
        "avg_monthly_revenue": avg_monthly_revenue,
        "intervention_cost": intervention_cost,
        "gross_benefit_q": benefit,
        "intervention_cost_total_q": cost,
        "net_impact_q": net
    }
    with open(os.path.join(out_dir, "business_impact.json"), "w") as f:
        json.dump(report, f, indent=2)
    mlflow.log_artifact(os.path.join(out_dir, "business_impact.json"))

    # log como métricas
    mlflow.log_metric("bi_topk", k)
    mlflow.log_metric("bi_true_churn_in_topk", num_true_churn_targeted)
    mlflow.log_metric("bi_retained_customers", retained)
    mlflow.log_metric("bi_net_impact_q", float(net))
    return report

# ========= main =========

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc_dir", default="../data/processed")
    ap.add_argument("--mlflow_uri", default="http://localhost:5000")
    ap.add_argument("--experiment", default="telco-churn")
    ap.add_argument("--run_id", default=None, help="RunID a evaluar; si no se pasa, toma el mejor por val_f1.")
    ap.add_argument("--model_stage", default="final_model", choices=["final_model","model"], help="Artifact dentro del run.")
    ap.add_argument("--n_splits", type=int, default=5, help="Folds para temporal CV.")
    # negocio
    ap.add_argument("--avg_monthly_revenue", type=float, default=850.0)
    ap.add_argument("--intervention_cost", type=float, default=25.0)
    ap.add_argument("--intervention_rate", type=float, default=0.30)
    ap.add_argument("--retention_success_rate", type=float, default=0.25)
    args = ap.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    client = MlflowClient()

    # Data
    (Xtr,ytr,dtr),(Xva,yva,dva),(Xte,yte,dte), num_cols, cat_cols, target = load_splits(args.proc_dir)

    # Selección de modelo
    if args.run_id is None:
        run_id = find_best_run_id(args.experiment, client)
    else:
        run_id = args.run_id

    model = load_model_from_run(run_id, model_stage=args.model_stage)

    # Nuevo run de evaluación
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name="evaluation_module"):
        mlflow.log_param("eval_source_run_id", run_id)
        mlflow.log_param("eval_model_stage", args.model_stage)

        out_dir = os.path.join("mlruns_artifacts", mlflow.active_run().info.run_id)
        os.makedirs(out_dir, exist_ok=True)

        # --- Test set ---
        yte_prob = model.predict_proba(Xte)[:,1]
        test_metrics, test_cm = eval_and_log_basic(yte, yte_prob, out_dir, prefix="test")

        # --- Temporal CV (train+val combinados) ---
        X_full = pd.concat([Xtr, Xva], axis=0)
        y_full = np.concatenate([ytr, yva])
        dates_full = pd.concat([dtr["JoinDate"], dva["JoinDate"]], axis=0) if "JoinDate" in dtr.columns else pd.Series(pd.Timestamp("2025-01-01"), index=X_full.index)
        df_tcv = temporal_cv(model, X_full, y_full, dates_full, n_splits=args.n_splits, out_dir=out_dir)

        # --- Importancia de variables (Permutation Importance) ---
        # OJO: necesitamos nombres post-OHE. Los extraemos del preprocessor si es posible.
        try:
            pre = model.named_steps["prep"]
            num_features = list(pre.transformers_[0][2]) if pre.transformers_ else num_cols
            ohe = pre.transformers_[1][1].named_steps["ohe"]
            cat_features = ohe.get_feature_names_out(pre.transformers_[1][2]).tolist()
            feature_names = num_features + cat_features
        except Exception:
            # fallback: columnas crudas
            feature_names = list(Xte.columns)

        # Permutation sobre test (más honesto)
        _ = permutation_importance_report(model, Xte, yte, feature_names, out_dir)

        # --- SHAP ---
        _ = shap_report(model, Xte.sample(min(100, len(Xte)), random_state=42), out_dir)

        # --- Error analysis ---
        fp, fn = error_analysis(yte, yte_prob, Xte, out_dir, topk=50)

        # --- Impacto de negocio ---
        bi = business_impact(
            y_true=yte, y_prob=yte_prob,
            avg_monthly_revenue=args.avg_monthly_revenue,
            intervention_cost=args.intervention_cost,
            intervention_rate=args.intervention_rate,
            retention_success_rate=args.retention_success_rate,
            out_dir=out_dir
        )

        # Resumen en consola
        print("== Test metrics ==")
        for k,v in test_metrics.items():
            print(f"{k}: {v:.4f}")
        print("== Business impact (net Q) ==", bi["net_impact_q"])

    print("[OK] Evaluación completa registrada en MLflow.")

if __name__ == "__main__":
    main()
