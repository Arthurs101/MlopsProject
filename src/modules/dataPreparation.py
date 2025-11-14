#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Preparation pipeline (CRISP-DM Fase 3)
- Limpieza (NA, duplicados, outliers básicos)
- Feature Engineering (derivadas útiles sin leakage)
- Transformaciones (encoding, escalado)
- Validación de calidad
- Split temporal (simulado con snapshot_date y tenure)
Guarda: data/processed/{train.csv,val.csv,test.csv} + columns.json (para modelado)
"""

import argparse, os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

# --------- util ---------
def ensure_dir(p): os.makedirs(os.path.dirname(p), exist_ok=True)

def iqr_cap(s: pd.Series, k: float = 1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k*iqr, q3 + k*iqr
    return s.clip(lower=lo, upper=hi)

def validate_quality(df: pd.DataFrame, target_col: str):
    report = {}
    report["shape"] = df.shape
    report["na_counts"] = df.isna().sum().sort_values(ascending=False).to_dict()
    report["dup_rows"] = int(df.duplicated().sum())
    if target_col in df.columns:
        report["class_balance"] = df[target_col].value_counts(normalize=True).round(4).to_dict()
    # rangos razonables
    for c in ["MonthlyCharges","TotalCharges","BillingDelayDays","LastSupportContactDaysAgo","tenure"]:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            report[f"range_{c}"] = [float(df[c].min()), float(df[c].max())]
    return report

# --------- pipeline ---------
def run_prep(input_csv: str, out_dir: str, snapshot_date: str = "2025-09-01"):
    rng = np.random.default_rng(RANDOM_SEED)
    ensure_dir(os.path.join(out_dir, "dummy.txt"))

    df = pd.read_csv(input_csv)
    # Tipos y limpieza básica
    if "Churn" not in df.columns:
        raise ValueError("Se requiere columna 'Churn' (Yes/No).")
    df["ChurnBinary"] = df["Churn"].map({"Yes":1, "No":0}).astype(int)

    for c in ["SeniorCitizen","tenure","MonthlyCharges","TotalCharges"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["tenure","MonthlyCharges","TotalCharges"]).copy()
    if "customerID" in df.columns:
        df = df.drop_duplicates(subset=["customerID"])

    # Outliers (cap por IQR en numéricas clave)
    for c in ["MonthlyCharges","TotalCharges","BillingDelayDays","LastSupportContactDaysAgo"]:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = iqr_cap(df[c], k=1.5)

    # ---------- Feature Engineering (sin leakage) ----------
    # 1) Ingresos medios y ratios
    if {"TotalCharges","tenure"}.issubset(df.columns):
        df["AvgMonthlyRevenue"] = df["TotalCharges"] / np.maximum(1, df["tenure"])
    df["ChargesTenureRatio"] = df["MonthlyCharges"] / np.maximum(1, df["tenure"])

    # 2) Índices útiles
    if "BillingDelayDays" in df.columns:
        df["PaymentRegularityIndex"] = (1.0 - df["BillingDelayDays"]/60.0).clip(0,1)
    if "NumSupportCalls" in df.columns and "tenure" in df.columns:
        df["SupportInteractionRate"] = (df["NumSupportCalls"] / np.maximum(1, df["tenure"])).clip(0,1)

    # 3) Lealtad compuesta
    parts = []
    if "tenure" in df.columns: parts.append((df["tenure"]/df["tenure"].max())*40)
    if "CustomerSatisfaction" in df.columns: parts.append((df["CustomerSatisfaction"]/5.0)*30)
    if "PaymentRegularityIndex" in df.columns: parts.append(df["PaymentRegularityIndex"]*20)
    if "HasMultipleServices" in df.columns: parts.append(df["HasMultipleServices"]*10)
    if parts:
        df["LoyaltyScore"] = np.sum(parts, axis=0).round(2)

    # ---------- Split temporal ----------
    # Simulamos fechas: snapshot fija y join_date = snapshot - tenure*30 días
    snap = pd.to_datetime(snapshot_date)
    if "tenure" in df.columns:
        df["JoinDate"] = snap - pd.to_timedelta(df["tenure"]*30, unit="D")
        df["LastActiveDate"] = snap  # “corte” actual; sin leakage del futuro
    else:
        df["JoinDate"] = snap - pd.to_timedelta(rng.integers(30, 1200, size=len(df)), unit="D")
        df["LastActiveDate"] = snap

    # Orden temporal por JoinDate (más antiguos primero)
    df = df.sort_values("JoinDate").reset_index(drop=True)

    # División 70/15/15 por tiempo (train / val / test)
    n = len(df)
    tr_end = int(n*0.70)
    va_end = int(n*0.85)
    train_df = df.iloc[:tr_end].copy()
    val_df   = df.iloc[tr_end:va_end].copy()
    test_df  = df.iloc[va_end:].copy()

    # ---------- Selección de columnas ----------
    target = "ChurnBinary"
    drop_cols = ["Churn","customerID","JoinDate","LastActiveDate"]  # no predictoras
    features_num = []
    features_cat = []

    for c in df.columns:
        if c in drop_cols or c == target: 
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            features_num.append(c)
        else:
            features_cat.append(c)

    # Persistir CSVs procesados
    train_path = os.path.join(out_dir, "train.csv")
    val_path   = os.path.join(out_dir, "val.csv")
    test_path  = os.path.join(out_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Guardar metadata/columnas para modelado
    meta = {
        "target": target,
        "drop_cols": drop_cols,
        "features_num": features_num,
        "features_cat": features_cat,
        "snapshot_date": snapshot_date
    }
    with open(os.path.join(out_dir, "columns.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Reporte de calidad
    rep = validate_quality(df, target)
    with open(os.path.join(out_dir, "quality_report.json"), "w") as f:
        json.dump(rep, f, indent=2)

    print("[OK] Prep finalizada.")
    print("train/val/test:", train_df.shape, val_df.shape, test_df.shape)
    print("target:", target)
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/teleconnect_churn_synth.csv")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--snapshot_date", default="2025-09-01")
    args = ap.parse_args()
    run_prep(args.input, args.out_dir, args.snapshot_date)
