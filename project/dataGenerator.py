#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Genera un dataset Telco Churn de 15,500 filas a partir de un CSV base:
- Preserva todas las filas originales
- Sintetiza el resto manteniendo distribuciones y coherencias
- Asigna 'Churn' a sintéticos de forma condicional con un modelo entrenado en el real
- Agrega features de negocio SIN leakage, útiles para análisis de churn

Uso:
  python dataGenerator.py \
    --input /mnt/data/telcoChurnBase.csv \
    --output data/raw/teleconnect_churn_15500.csv \
    --n 15500 \
    --seed 42
"""

import argparse, os, warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# ========== Utilidades base ==========

def ensure_dirs(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def coerce_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Tipos esperados en Telco
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # TotalCharges puede venir vacío -> drop
    if "TotalCharges" in df.columns:
        df = df.dropna(subset=["TotalCharges"])

    # Duplicados por customerID si existe
    if "customerID" in df.columns:
        df = df.drop_duplicates(subset=["customerID"])

    # Quitar filas con NaN críticos
    needed = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in df.columns]
    if needed:
        df = df.dropna(subset=needed)

    df = df.reset_index(drop=True)
    return df


def split_columns(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ("customerID",)]
    cat_cols = [c for c in df.columns if c not in num_cols + ["customerID"]]
    return num_cols, cat_cols


def sample_numeric(series: pd.Series, size: int, jitter: float, rng: np.random.Generator):
    base = rng.choice(series.values, size=size, replace=True)
    std = np.nanstd(series.values)
    noise = rng.normal(0.0, (std * jitter) if std > 0 else 0.0, size=size)
    lo, hi = np.nanmin(series.values), np.nanmax(series.values)
    return np.clip(base + noise, lo, hi)


def sample_categorical(series: pd.Series, size: int, rng: np.random.Generator):
    probs = series.value_counts(normalize=True)
    return rng.choice(probs.index.to_numpy(), size=size, p=probs.values)


# ========== Generación sintética (sin Churn) ==========

def make_synthetic_like(df_real: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    num_cols, cat_cols = split_columns(df_real)
    gen_cols = [c for c in df_real.columns if c not in ("customerID", "Churn")]  # no generar estas dos
    num_gen = [c for c in num_cols if c in gen_cols]
    cat_gen = [c for c in cat_cols if c in gen_cols]

    data = {}

    for col in num_gen:
        data[col] = sample_numeric(df_real[col], size=n, jitter=0.05, rng=rng)

    for col in cat_gen:
        data[col] = sample_categorical(df_real[col], size=n, rng=rng)

    syn = pd.DataFrame(data)

    # ID sintético
    syn.insert(0, "customerID", ["TC{:07d}".format(i) for i in range(1, n + 1)])

    # Coherencia básica: TotalCharges ≈ MonthlyCharges * tenure + ruido
    if {"MonthlyCharges", "tenure"}.issubset(syn.columns):
        noise = rng.normal(0, 5.0, size=n)
        syn["TotalCharges"] = np.maximum(
            0, syn["MonthlyCharges"] * np.maximum(1, syn["tenure"]) + noise
        )

    return syn


# ========== Modelo condicional para Churn ==========

def train_churn_model(df_real: pd.DataFrame, seed: int = 42):
    if "Churn" not in df_real.columns:
        raise ValueError("El CSV base debe contener la columna 'Churn'.")

    y = df_real["Churn"].map({"Yes": 1, "No": 0})
    X = df_real.drop(columns=["Churn", "customerID"], errors="ignore")
    X = pd.get_dummies(X, drop_first=True)

    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        min_samples_split=4,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(Xtr, ytr)
    return rf, X.columns  # columnas para alinear dummies


def assign_churn_to_synthetics(df_syn: pd.DataFrame, model, cols_ref, rng: np.random.Generator) -> pd.DataFrame:
    X_syn = pd.get_dummies(df_syn.drop(columns=["customerID"], errors="ignore"), drop_first=True)
    X_syn = X_syn.reindex(columns=cols_ref, fill_value=0)
    probs = model.predict_proba(X_syn)[:, 1]

    out = df_syn.copy()
    out["Churn"] = np.where(rng.random(len(probs)) < probs, "Yes", "No")

    # Balance suave si quedara muy sesgado (objetivo ~50/50 ±10%)
    churn_rate = (out["Churn"] == "Yes").mean()
    if churn_rate < 0.40 or churn_rate > 0.60:
        thr = 0.5
        step = 0.02
        for _ in range(25):
            pred_yes = (probs > thr).mean()
            if abs(pred_yes - 0.50) < 0.01:
                break
            thr += (-step if pred_yes > 0.50 else step)
        out["Churn"] = np.where(probs > thr, "Yes", "No")
    return out


# ========== Features de negocio SIN leakage ==========

def add_business_features(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    d = df.copy()

    # 1) BillingDelayDays (0–60), influenciado levemente por PaperlessBilling/PaymentMethod
    if "PaperlessBilling" in d.columns:
        paperless = d["PaperlessBilling"].astype(str).str.lower().isin(["yes", "true", "si", "sí"])
        base_delay = rng.integers(0, 35, size=len(d))
        extra = rng.integers(0, 26, size=len(d)) * (~paperless)
        d["BillingDelayDays"] = np.clip(base_delay + extra, 0, 60).astype(int)
    else:
        d["BillingDelayDays"] = rng.integers(0, 61, size=len(d))

    # 2) LastSupportContactDaysAgo (0–120)
    d["LastSupportContactDaysAgo"] = rng.integers(0, 121, size=len(d))

    # 3) ContractRenewalRemainingDays según Contract
    def renewal_days(contract):
        cs = str(contract).lower()
        if cs.startswith("month"):
            return int(rng.integers(0, 30))
        return int(rng.integers(0, 365))
    if "Contract" in d.columns:
        d["ContractRenewalRemainingDays"] = d["Contract"].map(renewal_days)
    else:
        d["ContractRenewalRemainingDays"] = rng.integers(0, 365, size=len(d)).astype(int)

    # 4) NumSupportCalls ~ Poisson (sesgo leve por problemas técnicos)
    # Creamos una proxy de "problemas técnicos" por tipo de InternetService
    tech_prob = np.where(
        d.get("InternetService", "DSL").astype(str).str.lower().isin(["fiber optic", "fibra", "fiber"]),
        0.35, 0.20
    )
    d["TechComplaint"] = (rng.random(len(d)) < tech_prob).astype(int)
    lam = 1.5 + d["TechComplaint"] * 1.2  # más soporte si hubo quejas
    d["NumSupportCalls"] = rng.poisson(lam=lam).astype(int)

    # 5) CustomerSatisfaction (1–5), penalizando señales negativas
    base_sat = rng.integers(2, 6, size=len(d))  # 2..5
    penalty = (
        (d["BillingDelayDays"] > 30).astype(int)
        + (d["NumSupportCalls"] >= 4).astype(int)
        + (d["LastSupportContactDaysAgo"] > 60).astype(int)
    )
    d["CustomerSatisfaction"] = np.clip(base_sat - penalty, 1, 5)

    # 6) HasMultipleServices (≥2)
    multi_cols = [c for c in ["PhoneService", "InternetService", "StreamingTV", "StreamingMovies"] if c in d.columns]
    if multi_cols:
        def bin_yes(s):
            s = s.astype(str).str.lower()
            return s.isin(["yes", "true", "si", "sí"]).astype(int)
        bin_sum = 0
        for c in multi_cols:
            if d[c].dtype == object:
                bin_sum += bin_yes(d[c])
            else:
                bin_sum += (d[c] > 0).astype(int)
        d["HasMultipleServices"] = (bin_sum >= 2).astype(int)
    else:
        d["HasMultipleServices"] = rng.integers(0, 2, size=len(d))

    # 7) AvgMonthlyRevenue coherente
    if {"TotalCharges", "tenure"}.issubset(d.columns):
        d["AvgMonthlyRevenue"] = d["TotalCharges"] / np.maximum(1, d["tenure"])
    elif "MonthlyCharges" in d.columns:
        d["AvgMonthlyRevenue"] = d["MonthlyCharges"]
    else:
        d["AvgMonthlyRevenue"] = rng.normal(70, 20, size=len(d))

    # 8) Índices derivados (0..1 o escala útil) sin usar la etiqueta
    d["PaymentRegularityIndex"] = 1.0 - (d["BillingDelayDays"] / 60.0)
    d["PaymentRegularityIndex"] = d["PaymentRegularityIndex"].clip(0, 1)

    d["SupportInteractionRate"] = d["NumSupportCalls"] / np.maximum(1, d.get("tenure", 1))
    d["SupportInteractionRate"] = d["SupportInteractionRate"].clip(0, 1.0)

    # Escala 0–100 ponderando señales de lealtad observables
    d["LoyaltyScore"] = (
        (d.get("tenure", 0) / (d.get("tenure", 0).max() if d.get("tenure", 0).max() > 0 else 1)) * 40
        + (d["CustomerSatisfaction"] / 5.0) * 30
        + (d["PaymentRegularityIndex"]) * 20
        + (d["HasMultipleServices"]) * 10
    ).round(2)

    # 9) Prob de respuesta a promo en función de satisfacción y pago regular
    p_promo = (0.10 + 0.35 * (d["CustomerSatisfaction"] / 5.0) + 0.25 * d["PaymentRegularityIndex"]).clip(0.05, 0.95)
    d["PromoResponse"] = (rng.random(len(d)) < p_promo).astype(int)

    # 10) NetPromoterScore 0..10 correlacionado con satisfacción (sin etiqueta)
    # mapeo básico + ruido
    nps_base = (d["CustomerSatisfaction"] - 1) * (10 / 4.0)  # 1->0, 5->10
    d["NetPromoterScore"] = np.clip(nps_base + rng.normal(0, 1.0, size=len(d)), 0, 10).round().astype(int)

    return d


# ========== Main ==========

def main():
    ap = argparse.ArgumentParser(description="Generar Telco Churn ~15.5k filas + features de negocio.")
    ap.add_argument("--input", default="/mnt/data/telcoChurnBase.csv", help="Ruta al CSV base Telco.")
    ap.add_argument("--output", default="data/raw/teleconnect_churn_15500.csv", help="Ruta de salida.")
    ap.add_argument("--n", type=int, default=15500, help="Tamaño total final (incluye filas originales).")
    ap.add_argument("--seed", type=int, default=42, help="Semilla RNG.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f">> Leyendo base: {args.input}")
    df0 = pd.read_csv(args.input)
    df0 = coerce_and_clean(df0)

    if "Churn" not in df0.columns:
        raise ValueError("El CSV base debe contener la columna 'Churn' (Yes/No).")

    base_n = len(df0)
    target_n = args.n
    if base_n > target_n:
        raise ValueError(f"La base tiene {base_n} filas > objetivo {target_n}. Reduce --n o limpia duplicados.")

    synth_n = target_n - base_n
    print(f">> Filas base: {base_n} | A sintetizar: {synth_n} | Objetivo: {target_n}")

    # Modelo condicional entrenado en el real
    print(">> Entrenando modelo condicional para Churn...")
    model, cols_ref = train_churn_model(df0, seed=args.seed)

    syn = pd.DataFrame()
    if synth_n > 0:
        print(">> Generando filas sintéticas (distribuciones similares a la base)...")
        syn = make_synthetic_like(df0, n=synth_n, rng=rng)

        print(">> Asignando 'Churn' a sintéticos de forma condicional...")
        syn = assign_churn_to_synthetics(syn, model, cols_ref, rng)

        # Limpiar rangos realistas
        if "MonthlyCharges" in syn.columns:
            syn["MonthlyCharges"] = np.clip(syn["MonthlyCharges"], 5, 200)
        if {"MonthlyCharges", "tenure"}.issubset(syn.columns):
            noise = rng.normal(0, 5.0, size=len(syn))
            syn["TotalCharges"] = np.maximum(
                0, syn["MonthlyCharges"] * np.maximum(1, syn["tenure"]) + noise
            )

    # Concatenar: primero base (con su churn real), luego sintéticos
    df_all = pd.concat([df0, syn], axis=0, ignore_index=True)

    print(">> Agregando features de negocio SIN leakage a todo el dataset...")
    df_all = add_business_features(df_all, rng)

    # Reordenar columnas: dejar ID y etiqueta al inicio
    cols = df_all.columns.tolist()
    ordered = []
    for c in ["customerID", "Churn"]:
        if c in cols:
            ordered.append(c)
    ordered += [c for c in cols if c not in ("customerID", "Churn")]
    df_all = df_all[ordered]

    # Exportar
    ensure_dirs(args.output)
    df_all.to_csv(args.output, index=False, encoding="utf-8")

    churn_rate = (df_all["Churn"] == "Yes").mean()
    print(">> Listo.")
    print(f">> Salida: {args.output}")
    print(f">> Shape final: {df_all.shape}")
    print(f">> Churn rate total: {churn_rate:.4f}")
    if "CustomerSatisfaction" in df_all.columns:
        print(">> CustomerSatisfaction (min/mean/max):",
              int(df_all["CustomerSatisfaction"].min()),
              round(float(df_all["CustomerSatisfaction"].mean()), 2),
              int(df_all["CustomerSatisfaction"].max()))
    if "LoyaltyScore" in df_all.columns:
        print(">> LoyaltyScore (min/mean/max):",
              round(float(df_all["LoyaltyScore"].min()), 2),
              round(float(df_all["LoyaltyScore"].mean()), 2),
              round(float(df_all["LoyaltyScore"].max()), 2))


if __name__ == "__main__":
    main()
