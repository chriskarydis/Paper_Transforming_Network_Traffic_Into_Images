#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import differential_evolution

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import joblib

SEED = 42
OUT_DIR = Path(__file__).parent / "lightgbm_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_CSV = Path(__file__).parent / "features.csv"

CNN_CLASSES = [
    "0day", "blackEnergy", "distcc_exec_backdoor",
    "hydra_ftp", "hydra_ssh", "java_rmi",
    "mirai", "netbios_ssn", "normal_browsing",
    "replayAttacks", "ruby_drb", "smtp",
    "tomcat", "unreallrcd", "vsftpd", "zeus",
]


def main():
    df = pd.read_csv(FEATURES_CSV)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feature_cols = [c for c in df.columns if c not in ("class", "binary", "split", "pcap")]
    df[feature_cols] = df[feature_cols].fillna(0)

    le = LabelEncoder()
    le.fit(CNN_CLASSES)
    joblib.dump(le, OUT_DIR / "label_encoder.pkl")
    with open(OUT_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    splits = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    for s, d in splits.items():
        print(f"{s}: {len(d)} rows")

    X_train = splits["train"][feature_cols].values
    y_bin_train = splits["train"]["binary"].values
    y_mul_train = le.transform(splits["train"]["class"].values)

    X_val = splits["val"][feature_cols].values
    y_bin_val = splits["val"]["binary"].values
    y_mul_val = le.transform(splits["val"]["class"].values)

    X_test = splits["test"][feature_cols].values
    y_bin_test = splits["test"]["binary"].values
    y_mul_test = le.transform(splits["test"]["class"].values)

    # ── Binary model ──────────────────────────────────────────
    print("\n[Binary] training on train split...")
    bin_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", lgb.LGBMClassifier(
            n_estimators=400, num_leaves=63, learning_rate=0.05,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
            min_child_samples=10, reg_alpha=0.05, reg_lambda=1.0,
            random_state=SEED, n_jobs=1, verbose=-1,
        ))
    ])
    bin_pipeline.fit(X_train, y_bin_train)
    joblib.dump(bin_pipeline, OUT_DIR / "binary_model.pkl")

    bin_val_proba = bin_pipeline.predict_proba(X_val)
    bin_val_f1 = f1_score(y_bin_val, bin_val_proba.argmax(1), average="macro")
    bin_val_acc = accuracy_score(y_bin_val, bin_val_proba.argmax(1))
    print(f"  Val: acc={bin_val_acc:.4f} macro_f1={bin_val_f1:.4f}")

    bin_test_proba = bin_pipeline.predict_proba(X_test)
    bin_test_f1 = f1_score(y_bin_test, bin_test_proba.argmax(1), average="macro")
    bin_test_acc = accuracy_score(y_bin_test, bin_test_proba.argmax(1))
    print(f"  Test: acc={bin_test_acc:.4f} macro_f1={bin_test_f1:.4f}")

    # ── Multi-class model ──────────────────────────────────────
    print("\n[Multi] training on train split (with SMOTE)...")
    min_class_count = pd.Series(y_mul_train).value_counts().min()
    k_neighbors = max(1, min(2, min_class_count - 1))
    print(f"  smallest train class count={min_class_count}  SMOTE k_neighbors={k_neighbors}")

    smote = SMOTE(random_state=SEED, k_neighbors=k_neighbors)
    mul_pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", smote),
        ("clf", lgb.LGBMClassifier(
            objective="multiclass", n_estimators=1000, num_leaves=127,
            learning_rate=0.02, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=5, min_child_samples=5, reg_alpha=0.1, reg_lambda=1.0,
            class_weight="balanced", random_state=SEED, n_jobs=1, verbose=-1,
        ))
    ])
    mul_pipeline.fit(X_train, y_mul_train)
    joblib.dump(mul_pipeline, OUT_DIR / "multi_model.pkl")

    mul_val_proba = mul_pipeline.predict_proba(X_val)
    n_classes = len(le.classes_)
    # predict_proba may have fewer columns if some classes are absent from train, align
    full_val_proba = np.zeros((len(X_val), n_classes))
    present_classes = mul_pipeline.named_steps["clf"].classes_
    for j, c in enumerate(present_classes):
        full_val_proba[:, c] = mul_val_proba[:, j]

    raw_val_f1 = f1_score(y_mul_val, full_val_proba.argmax(1), average="macro")
    print(f"  Val raw macro F1 (argmax): {raw_val_f1:.4f}")

    print("[Multi] Calibrating thresholds via differential evolution on VAL split...")
    def neg_macro_f1(thresholds):
        preds = (full_val_proba * thresholds).argmax(axis=1)
        return -f1_score(y_mul_val, preds, average="macro", zero_division=0)

    opt = differential_evolution(neg_macro_f1, bounds=[(0.1, 10.0)] * n_classes,
                                  seed=SEED, maxiter=300, tol=1e-4, polish=True)
    best_thresholds = opt.x
    np.save(OUT_DIR / "class_thresholds.npy", best_thresholds)

    val_pred_tuned = (full_val_proba * best_thresholds).argmax(axis=1)
    tuned_val_f1 = f1_score(y_mul_val, val_pred_tuned, average="macro")
    print(f"  Val macro F1 (calibrated): {tuned_val_f1:.4f}")

    # Test evaluation with calibrated thresholds
    mul_test_proba_raw = mul_pipeline.predict_proba(X_test)
    full_test_proba = np.zeros((len(X_test), n_classes))
    for j, c in enumerate(present_classes):
        full_test_proba[:, c] = mul_test_proba_raw[:, j]

    test_pred_raw = full_test_proba.argmax(axis=1)
    raw_test_f1 = f1_score(y_mul_test, test_pred_raw, average="macro")
    raw_test_acc = accuracy_score(y_mul_test, test_pred_raw)

    test_pred_tuned = (full_test_proba * best_thresholds).argmax(axis=1)
    tuned_test_f1 = f1_score(y_mul_test, test_pred_tuned, average="macro")
    tuned_test_acc = accuracy_score(y_mul_test, test_pred_tuned)
    print(f"  Test macro F1 (raw):        {raw_test_f1:.4f}  acc={raw_test_acc:.4f}")
    print(f"  Test macro F1 (calibrated): {tuned_test_f1:.4f}  acc={tuned_test_acc:.4f}")

    np.savez(OUT_DIR / "val_predictions.npz", proba=full_val_proba, labels=y_mul_val,
             bin_proba=bin_val_proba, bin_labels=y_bin_val)
    np.savez(OUT_DIR / "test_predictions.npz", proba=full_test_proba, labels=y_mul_test,
             bin_proba=bin_test_proba, bin_labels=y_bin_test,
             proba_calibrated=full_test_proba * best_thresholds / (full_test_proba * best_thresholds).sum(axis=1, keepdims=True))

    results = {
        "binary":  {"val_acc": bin_val_acc, "val_f1": bin_val_f1,
                    "test_acc": bin_test_acc, "test_f1": bin_test_f1},
        "multiclass": {"val_f1_raw": raw_val_f1, "val_f1_calibrated": tuned_val_f1,
                        "test_f1_raw": raw_test_f1, "test_acc_raw": raw_test_acc,
                        "test_f1_calibrated": tuned_test_f1, "test_acc_calibrated": tuned_test_acc},
        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
