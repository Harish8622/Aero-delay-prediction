# ml_pipelines/training/train_time_split_xgb.py
import argparse
import os
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


# ---------- helpers ----------
def pick_feature_columns(df):
    """Choose from columns that actually exist."""
    cat_candidates = ["AIRLINE", "ORIGIN", "DEST"]
    num_candidates = [
        "DISTANCE", "day_of_week", "month", "hour_of_day", "is_bank_holiday",
        "dep_rain", "dep_ice", "dep_wind", "arr_rain", "arr_ice", "arr_wind",
        # include continuous weather if you added them later:
        # "dep_temp", "dep_wspd", "dep_prcp", "arr_temp", "arr_wspd", "arr_prcp"
    ]
    cat_cols = [c for c in cat_candidates if c in df.columns]
    num_cols = [c for c in num_candidates if c in df.columns]
    return cat_cols, num_cols


def build_preprocessor(cat_cols, num_cols):
    """OneHot (sparse) for categoricals + passthrough numerics."""
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


def best_f1_threshold(y_true, proba):
    """Simple sweep over thresholds to find best F1."""
    thresholds = np.linspace(0.1, 0.9, 17)
    scores = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        scores.append((t, f1_score(y_true, pred)))
    t_best, f1_best = max(scores, key=lambda x: x[1])
    return t_best, f1_best


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Train XGBoost with time-based split (train=2022, test=2023+).")
    parser.add_argument("--data", type=str, default="test_date/data/processed/preprocessed_data.csv")
    parser.add_argument("--target", type=str, default="DELAY_FLAG_30")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=800)  # adjust as you like
    parser.add_argument("--learning_rate", type=float, default=0.06)
    parser.add_argument("--max_depth", type=int, default=7)
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    # ---- Load
    print(f"Loading {args.data} ...")
    df = pd.read_csv(args.data)

    # ---- Validate date column
    if "FL_DATE" not in df.columns:
        raise ValueError(
            "FL_DATE column not found. Re-run preprocessing to keep FL_DATE for time-based split, "
            "or split by date during preprocessing and load separate train/test files."
        )
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
    df = df.dropna(subset=["FL_DATE"])
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found in columns.")

    # ---- Time split
    train_mask = (df["FL_DATE"] >= "2022-01-01") & (df["FL_DATE"] <= "2022-12-31")
    test_mask  = (df["FL_DATE"] >= "2023-01-01")
    df_train = df.loc[train_mask].copy()
    df_test  = df.loc[test_mask].copy()

    if df_train.empty or df_test.empty:
        raise ValueError(f"Time split produced empty sets: train={len(df_train)}, test={len(df_test)}. Check FL_DATE values.")

    # ---- Features
    cat_cols, num_cols = pick_feature_columns(df)
    features = cat_cols + num_cols
    keep_cols = features + [args.target, "FL_DATE"]

    df_train = df_train[keep_cols].dropna(subset=[args.target])
    df_test  = df_test[keep_cols].dropna(subset=[args.target])

    y_train = df_train[args.target].astype(int).values
    y_test  = df_test[args.target].astype(int).values
    X_train = df_train[features]
    X_test  = df_test[features]

    print("Features:")
    print("  categoricals:", cat_cols)
    print("  numerics    :", num_cols)
    print("Class balance TRAIN:", Counter(y_train))
    print("Class balance TEST :", Counter(y_test))

    # ---- Preprocess (fit on TRAIN only)
    preprocess = build_preprocessor(cat_cols, num_cols)
    print("Fitting encoder on TRAIN ...")
    preprocess.fit(X_train)
    print("Transforming ...")
    Xtr_enc = preprocess.transform(X_train)
    Xte_enc = preprocess.transform(X_test)

    # ---- Class imbalance handling (no undersampling)
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = float(neg) / max(float(pos), 1.0)
    print(f"scale_pos_weight = {scale_pos_weight:.3f}  (neg={neg}, pos={pos})")

    # ---- Model (no early stopping; print progress every 50 rounds)
    xgb = XGBClassifier(
        objective="binary:logistic",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2.0,
        reg_lambda=2.0,
        reg_alpha=0.1,
        gamma=0.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=args.seed,
        verbosity=1,
    )

    eval_set = [(Xtr_enc, y_train), (Xte_enc, y_test)]
    print(f"Training XGBoost ({args.n_estimators} trees) with progress every 50 rounds...")
    xgb.fit(
        Xtr_enc, y_train,
        eval_set=eval_set,
        verbose=50  # prints eval metric each 50 boosting rounds
    )
    print("Training done.")

    # ---- Evaluate at default threshold 0.5
    proba = xgb.predict_proba(Xte_enc)[:, 1]
    pred05 = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy@0.5": float(accuracy_score(y_test, pred05)),
        "f1@0.5": float(f1_score(y_test, pred05)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
    }
    print("\nMetrics (threshold=0.5):", metrics)
    print("\nClassification report @0.5:\n", classification_report(y_test, pred05, digits=3))
    print("Confusion matrix @0.5:\n", confusion_matrix(y_test, pred05))

    # ---- Simple threshold sweep for best F1
    t_best, f1_best = best_f1_threshold(y_test, proba)
    pred_best = (proba >= t_best).astype(int)
    print(f"\nBest F1 threshold ~ {t_best:.2f}  (F1={f1_best:.3f})")
    print("Confusion matrix @best-F1:\n", confusion_matrix(y_test, pred_best))
    print("Classification report @best-F1:\n", classification_report(y_test, pred_best, digits=3))

    # ---- Save artifacts
    preproc_path = os.path.join(args.model_dir, "preprocess_time_split.pkl")
    model_path = os.path.join(args.model_dir, "xgb_time_split.json")
    joblib.dump(preprocess, preproc_path)
    xgb.save_model(model_path)
    print(f"\nSaved:\n  {preproc_path}\n  {model_path}")


if __name__ == "__main__":
    main()