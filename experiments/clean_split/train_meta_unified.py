#!/usr/bin/env python3
"""Trains the Meta-LightGBM stacker on the unified VAL split's stacked
base-model probabilities (never seeing test data) and evaluates it on the
unified TEST split, directly comparable to WSV/MCV/MinCV in Table VI."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (f1_score, accuracy_score, precision_score, recall_score,
                              roc_auc_score, matthews_corrcoef)
import lightgbm as lgb

EXP_DIR = Path(__file__).parent
SEED = 42

LGBM_PARAMS_BIN = dict(
    objective="binary", n_estimators=500, learning_rate=0.05, num_leaves=31,
    max_depth=5, min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, random_state=SEED, n_jobs=-1, verbose=-1,
)
LGBM_PARAMS_MUL = dict(
    objective="multiclass", n_estimators=500, learning_rate=0.05, num_leaves=31,
    max_depth=5, min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, random_state=SEED, n_jobs=-1, verbose=-1,
)


def metrics_multiclass(y_true, P):
    y_pred = P.argmax(1)
    return dict(
        acc=accuracy_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred, average="macro"),
        prec=precision_score(y_true, y_pred, average="macro", zero_division=0),
        rec=recall_score(y_true, y_pred, average="macro", zero_division=0),
        mcc=matthews_corrcoef(y_true, y_pred),
        auc=roc_auc_score(y_true, P, multi_class="ovr", average="macro",
                           labels=list(range(P.shape[1]))),
    )


def metrics_binary(y_true, P):
    y_pred = P.argmax(1)
    return dict(
        acc=accuracy_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred, average="macro"),
        mcc=matthews_corrcoef(y_true, y_pred),
        auc=roc_auc_score(y_true, P[:, 1]),
    )


def main():
    val = np.load(EXP_DIR / "meta_features_val.npz", allow_pickle=True)
    test = np.load(EXP_DIR / "meta_features_test.npz", allow_pickle=True)

    y_mul_val, y_mul_test = val["labels"], test["labels"]
    CNN_CLASSES_BENIGN_IDX = 8  # normal_browsing index, matches CNN_CLASSES order used throughout
    y_bin_val = (y_mul_val != CNN_CLASSES_BENIGN_IDX).astype(int)
    y_bin_test = (y_mul_test != CNN_CLASSES_BENIGN_IDX).astype(int)

    print(f"Train (val split): {val['X_mul'].shape}  Eval (test split): {test['X_mul'].shape}")

    # ---- Multiclass meta-learner ----
    mul_model = lgb.LGBMClassifier(**LGBM_PARAMS_MUL, num_class=16)
    mul_model.fit(val["X_mul"], y_mul_val)
    P_mul_test = mul_model.predict_proba(test["X_mul"])
    mul_metrics = metrics_multiclass(y_mul_test, P_mul_test)
    print("Multiclass test metrics:", {k: round(v, 4) for k, v in mul_metrics.items()})

    # ---- Binary meta-learner ----
    bin_model = lgb.LGBMClassifier(**LGBM_PARAMS_BIN)
    bin_model.fit(val["X_bin"], y_bin_val)
    P_bin_test = bin_model.predict_proba(test["X_bin"])
    bin_metrics = metrics_binary(y_bin_test, P_bin_test)
    print("Binary test metrics:", {k: round(v, 4) for k, v in bin_metrics.items()})

    out = {"multiclass": mul_metrics, "binary": bin_metrics,
           "n_val_train": int(len(y_mul_val)), "n_test_eval": int(len(y_mul_test))}
    json.dump(out, open(EXP_DIR / "meta_lightgbm_unified_results.json", "w"), indent=2)
    print(f"\nSaved {EXP_DIR / 'meta_lightgbm_unified_results.json'}")


if __name__ == "__main__":
    main()
