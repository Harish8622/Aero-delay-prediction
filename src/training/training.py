# ml_pipelines/training/train_xgb_search.py
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
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


def pick_feature_columns(df):
    """Choose features present in the dataframe."""
    cat_candidates = ["AIRLINE", "ORIGIN", "DEST", "ROUTE"]  # add ROUTE if you engineered it
    num_candidates = [
        "DISTANCE", "day_of_week", "month", "hour_of_day", "is_bank_holiday",
        "dep_rain", "dep_ice", "dep_wind", "arr_rain", "arr_ice", "arr_wind"
    ]
    cat_cols = [c for c in cat_candidates if c in df.columns]
    num_cols = [c for c in num_candidates if c in df.columns]
    return cat_cols, num_cols


def build_preprocessor(cat_cols, num_cols):
    """OneHot for cats (sparse), passthrough for nums."""
    # Keep compatible with older sklearn that uses 'sparse' instead of 'sparse_output'
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
    parser = argparse.ArgumentParser(description="Train XGBoost with light param search (AUC-PR).")
    parser.add_argument("--data", type=str, default="test_date/data/processed/preprocessed_data.csv")
    parser.add_argument("--target", type=str, default="DELAY_FLAG_15")
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--search_iters", type=int, default=20, help="Randomized search iterations")
    parser.add_argument("--cv_folds", type=int, default=3, help="CV folds for search")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    # -------- Load
    print(f"Loading {args.data} ...")
    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found in columns.")

    cat_cols, num_cols = pick_feature_columns(df)
    features = cat_cols + num_cols
    keep = features + [args.target]
    df = df[keep].dropna()

    y = df[args.target].astype(int)
    X = df[features]

    print("Features:")
    print("  categoricals:", cat_cols)
    print("  numerics    :", num_cols)
    print("Class balance (all):", Counter(y))

    # -------- Split (train/test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    # -------- Preprocess
    preprocess = build_preprocessor(cat_cols, num_cols)
    print("Fitting encoder on TRAIN ...")
    preprocess.fit(X_train)
    print("Transforming ...")
    Xtr_enc = preprocess.transform(X_train)
    Xte_enc = preprocess.transform(X_test)

    # -------- Class weights (use all data; no undersampling)
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / max(pos, 1)
    print(f"scale_pos_weight = {scale_pos_weight:.3f}  (neg={neg}, pos={pos})")

    # -------- Base model
    base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=args.seed,
        # fixed sensible defaults; search will override some
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.1,
        gamma=0.0,
        scale_pos_weight=scale_pos_weight,
        verbosity=1,  # a bit of chatter
    )

    # -------- Param search space (small & effective)
    param_dist = {
        "n_estimators": [600, 900, 1200, 1800, 2400, 3000],
        "learning_rate": np.logspace(np.log10(0.015), np.log10(0.12), 8),
        "max_depth": [4, 5, 6, 7, 8],
        "min_child_weight": [1.0, 2.0, 3.0, 5.0, 8.0],
        "subsample": [0.65, 0.75, 0.85, 0.95],
        "colsample_bytree": [0.65, 0.75, 0.85, 0.95],
        "reg_lambda": [0.5, 1.0, 2.0, 3.0, 5.0, 8.0],
        "reg_alpha": [0.0, 0.05, 0.1, 0.25, 0.5],
        "gamma": [0.0, 0.2, 0.5, 1.0],
        # keep scale_pos_weight fixed from y_train
    }

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=args.search_iters,
        scoring="average_precision",   # AUC-PR
        n_jobs=-1,
        cv=cv,
        refit=True,                    # refit best on full TRAIN
        verbose=2,                     # prints progress
        random_state=args.seed,
    )

    print(f"\nStarting RandomizedSearchCV for {args.search_iters} iterations...")
    search.fit(Xtr_enc, y_train)

    print("\nBest CV AUC-PR:", search.best_score_)
    print("Best params:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    best_model = search.best_estimator_

    # -------- Evaluate on TEST
    print("\nEvaluating on TEST ...")
    proba = best_model.predict_proba(Xte_enc)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
    }
    print("\nMetrics:", metrics)
    print("\nClassification report:\n", classification_report(y_test, pred, digits=3))

    # -------- Save artifacts
    preproc_path = os.path.join(args.model_dir, "preprocess.pkl")
    model_path = os.path.join(args.model_dir, "xgb_model.json")
    joblib.dump(preprocess, preproc_path)
    best_model.save_model(model_path)

    print(f"\nSaved:\n  {preproc_path}\n  {model_path}")


if __name__ == "__main__":
    main()