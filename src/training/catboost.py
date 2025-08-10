# src/training/train_time_split_catboost.py
import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)


def pick_feature_columns(df):
    """Return (cat_cols, num_cols) that actually exist in df."""
    cat_candidates = ["AIRLINE", "ORIGIN", "DEST"]
    num_candidates = [
        "DISTANCE", "day_of_week", "month", "hour_of_day", "is_bank_holiday",
        "dep_rain", "dep_ice", "dep_wind", "arr_rain", "arr_ice", "arr_wind",
        # add continuous weather later if you have them:
        # "dep_temp", "dep_wspd", "dep_prcp", "arr_temp", "arr_wspd", "arr_prcp"
    ]
    cat_cols = [c for c in cat_candidates if c in df.columns]
    num_cols = [c for c in num_candidates if c in df.columns]
    return cat_cols, num_cols


def best_f1_threshold(y_true, proba):
    """Simple sweep to find the threshold that maximizes F1 on the test set."""
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        pred = (proba >= t).astype(int)
        f1 = f1_score(y_true, pred)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def main():
    parser = argparse.ArgumentParser(
        description="Train CatBoost (time-split: train=2022, test=2023+)."
    )
    parser.add_argument("--data", type=str, default="test_date/data/processed/preprocessed_data.csv")
    parser.add_argument("--target", type=str, default="DELAY_FLAG_30")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--learning_rate", type=float, default=0.06)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--l2_leaf_reg", type=float, default=3.0)
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    # ---- Load
    print(f"Loading {args.data} ...")
    df = pd.read_csv(args.data)

    # ---- Checks
    if "FL_DATE" not in df.columns:
        raise ValueError("FL_DATE is required for time-based split (2022 train, 2023+ test).")
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found in columns.")

    # ---- Time split
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
    df = df.dropna(subset=["FL_DATE"])
    train_mask = (df["FL_DATE"] >= "2019-01-01") & (df["FL_DATE"] <= "2022-12-31")
    test_mask  = (df["FL_DATE"] >= "2023-01-01")

    df_train = df.loc[train_mask].copy()
    df_test  = df.loc[test_mask].copy()
    if df_train.empty or df_test.empty:
        raise ValueError(f"Empty split: train={len(df_train)}, test={len(df_test)}. Check FL_DATE range.")

    # ---- Features
    cat_cols, num_cols = pick_feature_columns(df)
    features = cat_cols + num_cols
    keep_cols = features + [args.target, "FL_DATE"]

    df_train = df_train[keep_cols].dropna(subset=[args.target])
    df_test  = df_test[keep_cols].dropna(subset=[args.target])

    # CatBoost can handle NaN in numeric; fill missing categoricals with 'UNK'
    for c in cat_cols:
        df_train[c] = df_train[c].astype("string").fillna("UNK")
        df_test[c]  = df_test[c].astype("string").fillna("UNK")

    X_train = df_train[features]
    y_train = df_train[args.target].astype(int).values
    X_test  = df_test[features]
    y_test  = df_test[args.target].astype(int).values

    print("Features:")
    print("  categoricals:", cat_cols)
    print("  numerics    :", num_cols)
    print("Class balance TRAIN:", Counter(y_train))
    print("Class balance TEST :", Counter(y_test))

    # Indices of categorical columns for CatBoost Pool
    cat_idx = [features.index(c) for c in cat_cols]

    # ---- Pools
    train_pool = Pool(X_train, label=y_train, cat_features=cat_idx)
    test_pool  = Pool(X_test,  label=y_test,  cat_features=cat_idx)

    # ---- Imbalance handling
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    spw = float(neg) / max(float(pos), 1.0)
    print(f"scale_pos_weight = {spw:.3f}  (neg={neg}, pos={pos})")

    # ---- Model
    model = CatBoostClassifier(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        loss_function="Logloss",
        eval_metric="AUC:use_weights=false",   # prints ROC-AUC; PR-AUC we’ll compute after
        random_seed=args.seed,
        scale_pos_weight=spw,
        thread_count=-1,
        bootstrap_type="Bayesian",
        verbose=50,               # progress every 50 iters
        task_type="CPU",          # change to "GPU" if you have one
    )

    print(f"Training CatBoost for {args.iterations} iterations...")
    model.fit(train_pool, eval_set=test_pool, use_best_model=False)
    print("Training done.")

    # ---- Evaluate
    proba = model.predict_proba(test_pool)[:, 1]
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

    # Threshold sweep for best F1
    t_best, f1_best = best_f1_threshold(y_test, proba)
    pred_best = (proba >= t_best).astype(int)
    print(f"\nBest F1 threshold ~ {t_best:.2f} (F1={f1_best:.3f})")
    print("Confusion matrix @best-F1:\n", confusion_matrix(y_test, pred_best))
    print("Classification report @best-F1:\n", classification_report(y_test, pred_best, digits=3))

    # ---- Save artifacts
    model_path = os.path.join(args.model_dir, "catboost_time_split.cbm")
    meta_path  = os.path.join(args.model_dir, "catboost_time_split.meta.json")
    os.makedirs(args.model_dir, exist_ok=True)
    model.save_model(model_path)

    with open(meta_path, "w") as f:
        json.dump(
            {
                "features": features,
                "categorical_features": cat_cols,
                "numeric_features": num_cols,
                "target": args.target,
                "train_range": ["2019-01-01", "2022-12-31"],
                "test_range":  ["2023-01-01", None],
                "scale_pos_weight": spw,
                "params": dict(
                    iterations=args.iterations,
                    learning_rate=args.learning_rate,
                    depth=args.depth,
                    l2_leaf_reg=args.l2_leaf_reg,
                    random_seed=args.seed,
                ),
            },
            f,
            indent=2,
        )

    print(f"\nSaved:\n  {model_path}\n  {meta_path}")


if __name__ == "__main__":
    main()