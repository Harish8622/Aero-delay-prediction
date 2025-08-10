# ml_pipelines/training/train_xgb_quick.py
import argparse
import os
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


def pick_feature_columns(df):
    """Select feature columns that exist in the dataframe."""
    cat_candidates = ["AIRLINE", "ORIGIN", "DEST"]  # add "ROUTE" if you've created it
    num_candidates = [
        "DISTANCE", "day_of_week", "month", "hour_of_day", "is_bank_holiday",
        "dep_rain", "dep_ice", "dep_wind", "arr_rain", "arr_ice", "arr_wind"
    ]
    cat_cols = [c for c in cat_candidates if c in df.columns]
    num_cols = [c for c in num_candidates if c in df.columns]
    return cat_cols, num_cols


def build_preprocessor(cat_cols, num_cols):
    """OneHot for categoricals (sparse), passthrough for numerics."""
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

    return ColumnTransformer(
        transformers=[
            ("cat", ohe, cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )


def main():
    parser = argparse.ArgumentParser(description="Quick baseline XGBoost (400 trees) with progress.")
    parser.add_argument("--data", type=str, default="test_date/data/processed/preprocessed_data.csv")
    parser.add_argument("--target", type=str, default="DELAY_FLAG_30")
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_dir", type=str, default="models")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    # ---- Load
    print(f"Loading {args.data} ...")
    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found in columns.")

    cat_cols, num_cols = pick_feature_columns(df)
    features = cat_cols + num_cols
    df = df[features + [args.target]].dropna()

    y = df[args.target].astype(int)
    X = df[features]

    print("Features:")
    print("  categoricals:", cat_cols)
    print("  numerics    :", num_cols)
    print("Class balance (all):", Counter(y))

    # ---- Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    # ---- Preprocess
    preprocess = build_preprocessor(cat_cols, num_cols)
    print("Fitting encoder on TRAIN ...")
    preprocess.fit(X_train)
    print("Transforming ...")
    Xtr_enc = preprocess.transform(X_train)
    Xte_enc = preprocess.transform(X_test)

    # ---- Class weight for imbalance
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)
    print(f"scale_pos_weight = {scale_pos_weight:.3f}  (neg={neg}, pos={pos})")

    # ---- Model (no early stopping, 400 trees) + live progress
    xgb = XGBClassifier(
        objective="binary:logistic",
        n_estimators=400,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2.0,
        reg_lambda=2.0,
        reg_alpha=0.1,
        gamma=0.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",   # AUC-PR is good for imbalanced data
        tree_method="hist",
        n_jobs=-1,
        random_state=args.seed,
        verbosity=1,
    )

    eval_set = [(Xtr_enc, y_train), (Xte_enc, y_test)]
    print("Training XGBoost (400 trees) with progress every 25 rounds...")
    xgb.fit(
        Xtr_enc, y_train,
        eval_set=eval_set,
        verbose=25  # print metrics every 25 boosting rounds
    )
    print("Done.")

    # ---- Evaluate
    proba = xgb.predict_proba(Xte_enc)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
    }
    print("\nMetrics:", metrics)
    print("\nClassification report:\n", classification_report(y_test, pred, digits=3))

    # ---- Save artifacts
    preproc_path = os.path.join(args.model_dir, "preprocess.pkl")
    model_path = os.path.join(args.model_dir, "xgb_quick_400.json")
    joblib.dump(preprocess, preproc_path)
    xgb.save_model(model_path)

    print(f"\nSaved:\n  {preproc_path}\n  {model_path}")


if __name__ == "__main__":
    main()