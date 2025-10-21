#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modeling con MLflow (CRISP-DM Fases 4 y 5)
- 4 modelos: LogisticRegression, RandomForest, XGBoost*, LightGBM*
- Tuning con RandomizedSearchCV
- Log de métricas (Precision, Recall, F1, ROC-AUC), curva ROC/PR, matriz de confusión
- Log de importancias y modelo
- Registro en Model Registry (opcional)
* Si no están instalados, se omiten y se escribe una nota en MLflow.
"""

import os, json, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

import matplotlib
#forcing headless since no graphic lib for now
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Opcionales
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

RANDOM_SEED = 42

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
        return X, y
    Xtr, ytr = split_xy(train)
    Xva, yva = split_xy(val)
    Xte, yte = split_xy(test)
    return (Xtr,ytr),(Xva,yva),(Xte,yte), num_cols, cat_cols

def build_preprocess(num_cols, cat_cols):
    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )
    return pre

def eval_and_log(y_true, y_prob, run_dir: str, prefix: str):
    os.makedirs(run_dir, exist_ok=True)  # <-- asegura carpeta del run
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

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{prefix} - Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for (i,j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha="center", va="center")
    fig.colorbar(im); fig.tight_layout()
    cm_path = os.path.join(run_dir, f"{prefix}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=120); plt.close()
    mlflow.log_artifact(cm_path)

    # ROC curve
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title(f"{prefix} - ROC Curve")
    fig.tight_layout()
    roc_path = os.path.join(run_dir, f"{prefix}_roc.png")
    os.makedirs(os.path.dirname(roc_path), exist_ok=True)
    plt.savefig(roc_path, dpi=120); plt.close()
    mlflow.log_artifact(roc_path)

    # PR curve
    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title(f"{prefix} - Precision-Recall Curve")
    fig.tight_layout()
    pr_path = os.path.join(run_dir, f"{prefix}_pr.png")
    os.makedirs(os.path.dirname(pr_path), exist_ok=True)
    plt.savefig(pr_path, dpi=120); plt.close()
    mlflow.log_artifact(pr_path)

    return metrics

def train_one(model_name: str, estimator, param_dist, Xtr, ytr, Xva, yva, preprocessor, experiment_name="telco-churn"):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model", model_name)

        pipe = Pipeline([
            ("prep", preprocessor),
            ("clf", estimator)
        ])

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=15,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_SEED,
            verbose=0
        )
        search.fit(Xtr, ytr)
        best = search.best_estimator_
        # Log hyperparams
        for k,v in search.best_params_.items():
            mlflow.log_param(k, v)

        # Eval train/val
        Path("mlruns_artifacts").mkdir(exist_ok=True)
        run_dir = os.path.join("mlruns_artifacts", mlflow.active_run().info.run_id)

        ytr_prob = best.predict_proba(Xtr)[:,1]
        yva_prob = best.predict_proba(Xva)[:,1]
        tr_metrics = eval_and_log(ytr, ytr_prob, run_dir, prefix="train")
        va_metrics = eval_and_log(yva, yva_prob, run_dir, prefix="val")

        # Importancias si existen
        try:
            clf = best.named_steps["clf"]
            if hasattr(clf, "feature_importances_"):
                mlflow.log_param("has_feature_importances_", True)
            else:
                mlflow.log_param("has_feature_importances_", False)
        except Exception:
            mlflow.log_param("has_feature_importances_", False)

        # Log del modelo
        mlflow.sklearn.log_model(best, artifact_path="model")

        # Métrica principal para comparar runs
        mlflow.log_metric("val_f1", va_metrics["f1"])

        return best, va_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc_dir", default="data/processed")
    ap.add_argument("--mlflow_uri", default="http://localhost:5000")
    ap.add_argument("--experiment", default="telco-churn")
    args = ap.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    (Xtr,ytr),(Xva,yva),(Xte,yte), num_cols, cat_cols = load_splits(args.proc_dir)
    pre = build_preprocess(num_cols, cat_cols)

    trained = []

    # 1) Logistic Regression
    lr = LogisticRegression(max_iter=1000, n_jobs=None)
    lr_space = {
        "clf__C": np.logspace(-3, 2, 20),
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"]
    }
    best_lr, m_lr = train_one("logreg", lr, lr_space, Xtr,ytr,Xva,yva, pre, args.experiment); trained.append(("logreg", best_lr, m_lr))

    # 2) Random Forest
    rf = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)
    rf_space = {
        "clf__n_estimators": [200,300,400,600],
        "clf__max_depth": [None, 8, 12, 16],
        "clf__min_samples_split": [2,4,6],
        "clf__min_samples_leaf": [1,2,4]
    }
    best_rf, m_rf = train_one("random_forest", rf, rf_space, Xtr,ytr,Xva,yva, pre, args.experiment); trained.append(("random_forest", best_rf, m_rf))

    # 3) XGBoost (si disponible)
    if HAS_XGB:
        xgb = XGBClassifier(
            random_state=RANDOM_SEED,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist"
        )
        xgb_space = {
            "clf__n_estimators": [200,400,600],
            "clf__max_depth": [3,4,6,8],
            "clf__learning_rate": [0.01,0.05,0.1,0.2],
            "clf__subsample": [0.7,0.9,1.0],
            "clf__colsample_bytree": [0.7,0.9,1.0]
        }
        best_xgb, m_xgb = train_one("xgboost", xgb, xgb_space, Xtr,ytr,Xva,yva, pre, args.experiment); trained.append(("xgboost", best_xgb, m_xgb))
    else:
        with mlflow.start_run(run_name="xgboost_skipped"):
            mlflow.log_param("model","xgboost")
            mlflow.log_param("skipped_reason","xgboost not installed")

    # 4) LightGBM (si disponible)
    if HAS_LGBM:
        lgbm = LGBMClassifier(random_state=RANDOM_SEED)
        lgbm_space = {
            "clf__n_estimators": [200,400,600],
            "clf__max_depth": [-1, 6, 10],
            "clf__learning_rate": [0.01,0.05,0.1],
            "clf__num_leaves": [31,63,127],
            "clf__subsample": [0.7,0.9,1.0]
        }
        best_lgbm, m_lgbm = train_one("lightgbm", lgbm, lgbm_space, Xtr,ytr,Xva,yva, pre, args.experiment); trained.append(("lightgbm", best_lgbm, m_lgbm))
    else:
        with mlflow.start_run(run_name="lightgbm_skipped"):
            mlflow.log_param("model","lightgbm")
            mlflow.log_param("skipped_reason","lightgbm not installed")

    # ---- Selección y evaluación final en test ----
    # Elige por mejor val_f1
    best_name, best_est, _ = sorted(trained, key=lambda t: t[2]["f1"], reverse=True)[0]

    with mlflow.start_run(run_name="final_test_eval"):
        mlflow.log_param("selected_model", best_name)
        # Reentrena en train+val (opcional, aquí lo ajustamos con train+val rápidamente)
        Xtrva = pd.concat([Xtr, Xva], axis=0)
        ytrva = np.concatenate([ytr, yva])
        best_est.fit(Xtrva, ytrva)

        yte_prob = best_est.predict_proba(Xte)[:,1]
        Path("mlruns_artifacts").mkdir(exist_ok=True)
        run_dir = os.path.join("mlruns_artifacts", mlflow.active_run().info.run_id)
        metrics_test = eval_and_log(yte, yte_prob, run_dir, prefix="test")

        mlflow.sklearn.log_model(best_est, artifact_path="final_model")
        mlflow.log_metric("test_f1", metrics_test["f1"])
        mlflow.log_metric("test_roc_auc", metrics_test["roc_auc"])

        # (Opcional) Registrar en Model Registry si tienes mlflow server con backend store
        # mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/final_model", "telco-churn-prod")

    print("[OK] Entrenamiento y evaluación registrados en MLflow.")
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc_dir", default="data/processed")
    ap.add_argument("--mlflow_uri", default="http://localhost:5000")
    ap.add_argument("--experiment", default="telco-churn")
    args = ap.parse_args()
    main()
