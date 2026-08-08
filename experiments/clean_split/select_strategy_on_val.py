#!/usr/bin/env python3
"""Selects the deployed fusion strategy and the LightGBM voter (calibrated vs.
raw) using ONLY the unified validation split, never the test split. This
supersedes the earlier practice of comparing strategies directly on the test
split (Table VI in the paper as originally written). Test-split numbers are
still computed and reported afterwards for the single selected configuration,
but they play no role in the selection itself.

Meta-LightGBM's validation score is estimated by 5-fold cross-validation
within the validation split (never in-sample on its own training data),
mirroring the CV protocol already used elsewhere in this work for
Meta-LightGBM's first-phase evaluation, just applied to validation instead
of test.
"""
import json
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

EXP_DIR = Path(__file__).parent
SEED = 42
MIN_CONF_THRESHOLD = 0.9
BENIGN_IDX = 8  # normal_browsing, CNN_CLASSES order used throughout


def load_branch_val():
    pkt = np.load(EXP_DIR / "models" / "Packet_Greyscale" / "val_predictions_aligned.npz", allow_pickle=True)
    flw = np.load(EXP_DIR / "models" / "Flow_RGB" / "val_predictions_aligned.npz", allow_pickle=True)
    lgb_npz = np.load(EXP_DIR / "lightgbm_model" / "val_predictions.npz", allow_pickle=True)
    thresholds = np.load(EXP_DIR / "lightgbm_model" / "class_thresholds.npy")

    assert np.array_equal(pkt["stems"], flw["stems"]), "Packet/Flow val order mismatch!"
    assert np.array_equal(pkt["labels"], lgb_npz["labels"]), "Packet/LightGBM val label mismatch!"

    y_true = pkt["labels"]
    P_pkt = pkt["mul_probs"]
    P_flw = flw["mul_probs"]

    P_lgb_raw = lgb_npz["proba"]
    P_lgb_cal = P_lgb_raw * thresholds
    P_lgb_cal /= P_lgb_cal.sum(axis=1, keepdims=True)

    pkt_summary = json.load(open(EXP_DIR / "models" / "Packet_Greyscale" / "summary.json"))
    flw_summary = json.load(open(EXP_DIR / "models" / "Flow_RGB" / "summary.json"))
    lgb_results = json.load(open(EXP_DIR / "lightgbm_model" / "results.json"))

    w1 = pkt_summary["best_val_mul_f1"]
    w2 = flw_summary["best_val_mul_f1"]
    w3_cal = lgb_results["multiclass"]["val_f1_calibrated"]
    w3_raw = lgb_results["multiclass"]["val_f1_raw"]

    return y_true, P_pkt, P_flw, P_lgb_raw, P_lgb_cal, w1, w2, w3_cal, w3_raw


def macro_f1(y_true, P):
    return f1_score(y_true, P.argmax(1), average="macro")


def wsv(P_pkt, P_flw, P_lgb, w1, w2, w3):
    w_tot = w1 + w2 + w3
    return (w1 * P_pkt + w2 * P_flw + w3 * P_lgb) / w_tot


def mcv(P_pkt, P_flw, P_lgb):
    n = P_pkt.shape[0]
    confs = np.stack([P_pkt.max(1), P_flw.max(1), P_lgb.max(1)], axis=1)
    winner = confs.argmax(1)
    stacked = np.stack([P_pkt, P_flw, P_lgb], axis=1)  # (n, 3, 16)
    return stacked[np.arange(n), winner, :]


def mincv(P_pkt, P_flw, P_lgb, w1, w2, w3, threshold=MIN_CONF_THRESHOLD):
    n = P_pkt.shape[0]
    out = np.zeros_like(P_pkt)
    branches = [(P_pkt, w1), (P_flw, w2), (P_lgb, w3)]
    for i in range(n):
        qual = [(P, w) for P, w in branches if P[i].max() >= threshold]
        if not qual:
            qual = branches
        w_tot = sum(w for _, w in qual)
        out[i] = sum(w * P[i] for P, w in qual) / w_tot
    return out


def meta_lightgbm_cv_val():
    """5-fold CV estimate of Meta-LightGBM's validation macro F1, out-of-fold
    (never trained and scored on the same rows)."""
    val = np.load(EXP_DIR / "meta_features_val.npz", allow_pickle=True)
    X, y = val["X_mul"], val["labels"]

    params = dict(
        objective="multiclass", n_estimators=500, learning_rate=0.05, num_leaves=31,
        max_depth=5, min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=SEED, n_jobs=-1, verbose=-1,
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_pred = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in skf.split(X, y):
        model = lgb.LGBMClassifier(**params, num_class=16)
        model.fit(X[train_idx], y[train_idx])
        oof_pred[test_idx] = model.predict(X[test_idx])
    return f1_score(y, oof_pred, average="macro")


def main():
    y_true, P_pkt, P_flw, P_lgb_raw, P_lgb_cal, w1, w2, w3_cal, w3_raw = load_branch_val()

    print("=== LightGBM voter selection (validation macro F1) ===")
    print(f"  raw:        {w3_raw:.4f}")
    print(f"  calibrated: {w3_cal:.4f}")
    lgb_winner = "calibrated" if w3_cal >= w3_raw else "raw"
    print(f"  -> selected: {lgb_winner}\n")

    P_lgb = P_lgb_cal if lgb_winner == "calibrated" else P_lgb_raw
    w3 = w3_cal if lgb_winner == "calibrated" else w3_raw

    print("=== Fusion strategy selection (validation macro F1, n=810) ===")
    f1_wsv = macro_f1(y_true, wsv(P_pkt, P_flw, P_lgb, w1, w2, w3))
    f1_mcv = macro_f1(y_true, mcv(P_pkt, P_flw, P_lgb))
    f1_mincv = macro_f1(y_true, mincv(P_pkt, P_flw, P_lgb, w1, w2, w3))
    f1_meta = meta_lightgbm_cv_val()

    results = {
        "Weighted Soft-Vote": f1_wsv,
        "Min-Confidence Vote": f1_mincv,
        "Max-Confidence Vote": f1_mcv,
        "Meta-LightGBM (5-fold CV within val)": f1_meta,
    }
    for name, f1 in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {name:40s} {f1:.4f}")

    winner = max(results, key=results.get)
    print(f"\n  -> selected strategy: {winner}")

    out = {
        "lgb_voter_selection": {"raw_val_f1": w3_raw, "calibrated_val_f1": w3_cal, "selected": lgb_winner},
        "strategy_selection_val_f1": results,
        "selected_strategy": winner,
        "weights_used": {"w1_packet": w1, "w2_flow": w2, "w3_lgbm": w3},
    }
    with open(EXP_DIR / "strategy_selection_on_val.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {EXP_DIR / 'strategy_selection_on_val.json'}")


if __name__ == "__main__":
    main()
