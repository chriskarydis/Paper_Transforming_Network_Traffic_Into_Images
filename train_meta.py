#!/usr/bin/env python3
"""
train_meta.py

Stacking meta-learner: trains a LightGBM model on top of the 6 CNN base
models + LightGBM base model, learning to optimally combine their predictions.

Input:  models/ensemble_eval/meta_features.npz
        (produced by evaluate_ensemble.py - contains full softmax vectors
         from all 6 CNNs and LightGBM for every test sample)

Output: models/meta/binary_meta.pkl    - LightGBM binary meta-classifier
        models/meta/multi_meta.pkl     - LightGBM multiclass meta-classifier
        models/meta/meta_results.csv   - per-fold and final metrics
        models/meta/metrics_table_binary.png
        models/meta/metrics_table_multiclass.png

Meta-features (per sample):
  Binary task:    6 CNNs × 2 probs  + LightGBM × 2 probs  = 14 features
  Multiclass task: 6 CNNs × 16 probs + LightGBM × 16 probs = 112 features

Training strategy: 5-fold stratified CV on the test set
  (base models never trained on the test set, so no data leakage)
  Final models are retrained on all samples after CV.
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
warnings.filterwarnings("ignore", message="The least populated class")
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, roc_auc_score,
)
import lightgbm as lgb

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE    = Path.home() / "thesis_heifip"
NPZ     = BASE / "models" / "ensemble_eval" / "meta_features.npz"
OUT_DIR = BASE / "models" / "meta"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CNN_CLASSES = [
    "0day", "blackEnergy", "distcc_exec_backdoor",
    "hydra_ftp", "hydra_ssh", "java_rmi",
    "mirai", "netbios_ssn", "normal_browsing",
    "replayAttacks", "ruby_drb", "smtp",
    "tomcat", "unreallrcd", "vsftpd", "zeus",
]
N_CLASSES = len(CNN_CLASSES)
N_FOLDS_BIN = 5   # binary: all classes have enough samples
N_FOLDS_MUL = 3   # multiclass: some rare classes have only 2 test samples
SEED        = 42

# ─── LightGBM hyperparameters ─────────────────────────────────────────────────
LGBM_PARAMS_BIN = dict(
    objective       = "binary",
    metric          = "binary_logloss",
    n_estimators    = 500,
    learning_rate   = 0.05,
    num_leaves      = 31,
    max_depth       = 5,
    min_child_samples = 10,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    reg_alpha       = 0.1,
    reg_lambda      = 0.1,
    random_state    = SEED,
    n_jobs          = -1,
    verbose         = -1,
)

LGBM_PARAMS_MUL = dict(
    objective        = "multiclass",
    metric           = "multi_logloss",
    num_class        = N_CLASSES,
    n_estimators     = 500,
    learning_rate    = 0.05,
    num_leaves       = 31,
    max_depth        = 5,
    min_child_samples= 10,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 0.1,
    random_state     = SEED,
    n_jobs           = -1,
    verbose          = -1,
)

# ─── Load meta-features ───────────────────────────────────────────────────────
def load_features(npz_path: Path):
    data = np.load(npz_path)
    y_bin = data["true_binary"].astype(int)
    y_mul = data["true_mul"].astype(int)

    base_models = ["pkt_grey", "pkt_rgb", "flw_grey", "flw_rgb",
                   "comb_grey", "comb_rgb", "lgb"]

    bin_feat_names = [f"{m}_bin_p{i}" for m in base_models for i in range(2)]
    mul_feat_names = [f"{m}_mul_p{i}" for m in base_models for i in range(N_CLASSES)]

    # Use DataFrames so LightGBM carries proper feature names and never warns
    X_bin = pd.DataFrame(
        np.hstack([data[f"{m}_bin"] for m in base_models]),
        columns=bin_feat_names,
    )
    X_mul = pd.DataFrame(
        np.hstack([data[f"{m}_mul"] for m in base_models]),
        columns=mul_feat_names,
    )

    print(f"Loaded {len(y_bin)} samples from {npz_path.name}")
    print(f"  Binary    meta-features: {X_bin.shape[1]}")
    print(f"  Multiclass meta-features: {X_mul.shape[1]}")
    return X_bin, X_mul, y_bin, y_mul, bin_feat_names, mul_feat_names


# ─── Cross-validated training ─────────────────────────────────────────────────
def cv_train(X, y, params, task, n_folds, y_bin=None, bin_scores=None):
    """bin_scores: P(malicious) array used for AUC in multiclass task."""
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    KW = dict(average="macro", zero_division=0)

    fold_metrics = []
    oof_preds  = np.zeros(len(y), dtype=int)
    oof_scores = np.zeros(len(y))   # P(malicious) for AUC

    print(f"\n{'='*70}")
    print(f"  {task.upper()} - {n_folds}-fold CV  ({X.shape[1]} features, {len(y)} samples)")
    print(f"{'='*70}")

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)],
        )

        proba = model.predict_proba(X_va)
        preds = model.predict(X_va)
        oof_preds[va_idx] = preds

        if task == "binary":
            scores = proba[:, 1]
            auc    = roc_auc_score(y_va, scores)
        else:
            scores   = bin_scores[va_idx]
            auc      = roc_auc_score(y_bin[va_idx], scores)

        oof_scores[va_idx] = scores

        acc  = accuracy_score(y_va, preds)
        f1   = f1_score(y_va, preds, **KW)
        prec = precision_score(y_va, preds, **KW)
        rec  = recall_score(y_va, preds, **KW)
        mcc  = matthews_corrcoef(y_va, preds)

        fold_metrics.append(dict(fold=fold, acc=acc, f1=f1,
                                 prec=prec, rec=rec, mcc=mcc, auc=auc))
        print(f"  Fold {fold}  Acc={acc:.4f}  F1={f1:.4f}  "
              f"Prec={prec:.4f}  Rec={rec:.4f}  MCC={mcc:.4f}  AUC={auc:.4f}  "
              f"(trees={model.best_iteration_})")

    # OOF aggregate
    KW2 = dict(average="macro", zero_division=0)
    oof_acc  = accuracy_score(y, oof_preds)
    oof_f1   = f1_score(y, oof_preds, **KW2)
    oof_prec = precision_score(y, oof_preds, **KW2)
    oof_rec  = recall_score(y, oof_preds, **KW2)
    oof_mcc  = matthews_corrcoef(y, oof_preds)
    if task == "binary":
        oof_auc = roc_auc_score(y, oof_scores)
    else:
        oof_auc = roc_auc_score(y_bin, oof_scores)

    print(f"\n  OOF totals: Acc={oof_acc:.4f}  F1={oof_f1:.4f}  "
          f"Prec={oof_prec:.4f}  Rec={oof_rec:.4f}  "
          f"MCC={oof_mcc:.4f}  AUC={oof_auc:.4f}")

    oof_summary = dict(fold="OOF", acc=oof_acc, f1=oof_f1,
                       prec=oof_prec, rec=oof_rec, mcc=oof_mcc, auc=oof_auc)

    # Retrain on all data with the mean best_iteration across folds
    best_iters = [lgb.LGBMClassifier(**params).fit(
        X.iloc[tr], y[tr],
        eval_set=[(X.iloc[va], y[va])],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(-1)],
    ).best_iteration_
    for tr, va in kf.split(X, y)]
    # Actually retrain cleanly on all data
    final_params = {**params, "n_estimators": int(np.mean(best_iters)) or params["n_estimators"]}
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(X, y, callbacks=[lgb.log_evaluation(-1)])
    print(f"  Final model retrained on all {len(y)} samples "
          f"(n_estimators={final_params['n_estimators']})")

    return final_model, fold_metrics, oof_summary, oof_preds


# ─── Metrics table PNG (same style as evaluate_ensemble.py) ──────────────────
def save_comparison_table(rows: list[dict], out_path: Path, title: str):
    col_hdrs = ["Model", "Accuracy", "F1-score", "Precision", "Recall", "MCC", "ROC-AUC"]
    col_w    = [3.5, 1.5, 1.5, 1.5, 1.5, 1.3, 1.5]
    total_w  = sum(col_w)
    mk       = ["acc", "f1", "prec", "rec", "mcc", "auc"]

    df_m = pd.DataFrame(rows).set_index("model")
    best = {k: df_m[k].max() for k in mk}

    n_rows   = len(rows)
    cell_h   = 0.55
    header_h = 0.65
    fig_h    = header_h + n_rows * cell_h + 0.5

    fig = plt.figure(figsize=(13, fig_h))
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    xs = [0.0]
    for w in col_w:
        xs.append(xs[-1] + w)

    def cell(x0, x1, y0, h, fc, ec="#cccccc", lw=0.5):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y0), x1 - x0, h,
            boxstyle="square,pad=0", linewidth=lw,
            facecolor=fc, edgecolor=ec,
        ))

    hy0 = fig_h - header_h
    for j in range(len(col_hdrs)):
        cell(xs[j], xs[j+1], hy0, header_h, "#2c3e50", "#2c3e50")
        ax.text((xs[j]+xs[j+1])/2, hy0 + header_h/2, col_hdrs[j],
                ha="center", va="center", color="white", fontweight="bold", fontsize=9)

    ROW_COLORS = ["#f0f4f8", "#eaf4fb", "#e9f7ef", "#fef9e7"]
    for i, row in enumerate(rows):
        y0  = fig_h - header_h - (i+1) * cell_h
        mid = y0 + cell_h / 2
        gc  = ROW_COLORS[i % len(ROW_COLORS)]
        for j in range(len(col_hdrs)):
            cell(xs[j], xs[j+1], y0, cell_h, gc)
        ax.text((xs[0]+xs[1])/2, mid, row["model"],
                ha="center", va="center", fontsize=8.5)
        for k_idx, key in enumerate(mk):
            j   = k_idx + 1
            xc  = (xs[j] + xs[j+1]) / 2
            val = df_m.loc[row["model"], key]
            is_best = abs(val - best[key]) < 1e-9
            if is_best:
                cell(xs[j], xs[j+1], y0, cell_h, "#d6eaf8")
            ax.text(xc, mid, f"{val*100:.2f}%",
                    ha="center", va="center", fontsize=8.5,
                    fontweight="bold" if is_best else "normal")

    ax.axhline(hy0, color="#333333", linewidth=1.2)
    ax.axhline(0,   color="#333333", linewidth=1.2)
    ax.set_title(title, y=1.0, pad=4, fontsize=11, fontweight="bold")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out_path}")


# ─── Confusion matrix ─────────────────────────────────────────────────────────
def save_confusion_matrix(y_true, y_pred, labels, title, out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig_size = max(8, len(labels))
    plt.figure(figsize=(fig_size, fig_size - 1))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=12, fontweight="bold", pad=10)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ─── Feature importance ───────────────────────────────────────────────────────
BASE_MODEL_NAMES = {
    "pkt_grey":  "Packet\nGreyscale",
    "pkt_rgb":   "Packet\nRGB",
    "flw_grey":  "Flow\nGreyscale",
    "flw_rgb":   "Flow\nRGB",
    "comb_grey": "Combined\nGreyscale",
    "comb_rgb":  "Combined\nRGB",
    "lgb":       "LightGBM",
}
BASE_MODELS = list(BASE_MODEL_NAMES.keys())


def save_feature_importance(model, feat_names: list[str], task: str, out_path: Path):
    """
    Two-panel figure:
      Left  - importance aggregated per base model (7 bars)
      Right - top-20 individual features
    """
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"feature": feat_names, "importance": importances})

    # Aggregate by base model (sum across all of that model's output dimensions)
    agg = {}
    for m in BASE_MODELS:
        mask = feat_df["feature"].str.startswith(m + "_")
        agg[BASE_MODEL_NAMES[m]] = feat_df.loc[mask, "importance"].sum()
    agg_df = pd.DataFrame(list(agg.items()), columns=["model", "importance"])
    agg_df = agg_df.sort_values("importance", ascending=True)

    # Top-20 individual features (shortened names)
    top20 = feat_df.nlargest(20, "importance").sort_values("importance", ascending=True)
    top20 = top20.copy()
    top20["label"] = top20["feature"].str.replace(r"_(bin|mul)_p", r" p", regex=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Meta-LightGBM Feature Importance - {task.capitalize()} Task",
                 fontsize=13, fontweight="bold", y=1.01)

    # Left: per-model aggregated importance
    colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(agg_df)))
    axes[0].barh(agg_df["model"], agg_df["importance"], color=colors, edgecolor="white")
    axes[0].set_xlabel("Total Importance (gain)")
    axes[0].set_title("Importance by Base Model", fontsize=11)
    axes[0].tick_params(axis="y", labelsize=9)
    for i, (_, row) in enumerate(agg_df.iterrows()):
        axes[0].text(row["importance"] + agg_df["importance"].max() * 0.01,
                     i, f"{row['importance']:.0f}", va="center", fontsize=8)

    # Right: top-20 individual features
    colors2 = plt.cm.Greens(np.linspace(0.35, 0.85, len(top20)))
    axes[1].barh(top20["label"], top20["importance"], color=colors2, edgecolor="white")
    axes[1].set_xlabel("Importance (gain)")
    axes[1].set_title("Top-20 Individual Features", fontsize=11)
    axes[1].tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    X_bin, X_mul, y_bin, y_mul, bin_names, mul_names = load_features(NPZ)

    # ── Binary meta-learner ───────────────────────────────────────────────────
    # P(malicious) from each base binary head, averaged - used for AUC
    p_malicious = X_bin.filter(like="_bin_p1").mean(axis=1).values

    bin_model, bin_folds, bin_oof, bin_oof_preds = cv_train(
        X_bin, y_bin, LGBM_PARAMS_BIN, "binary", N_FOLDS_BIN,
        y_bin=y_bin, bin_scores=p_malicious)
    joblib.dump(bin_model, OUT_DIR / "binary_meta.pkl")
    print(f"Saved {OUT_DIR / 'binary_meta.pkl'}")

    # ── Multiclass meta-learner ───────────────────────────────────────────────
    mul_model, mul_folds, mul_oof, mul_oof_preds = cv_train(
        X_mul, y_mul, LGBM_PARAMS_MUL, "multiclass", N_FOLDS_MUL,
        y_bin=y_bin, bin_scores=p_malicious)
    joblib.dump(mul_model, OUT_DIR / "multi_meta.pkl")
    print(f"Saved {OUT_DIR / 'multi_meta.pkl'}")

    # ── Confusion matrices (OOF predictions) ─────────────────────────────────
    save_confusion_matrix(
        y_bin, bin_oof_preds,
        labels=["benign", "malicious"],
        title="Meta-LightGBM - Binary Confusion Matrix (OOF)",
        out_path=OUT_DIR / "confusion_binary.png",
    )
    save_confusion_matrix(
        y_mul, mul_oof_preds,
        labels=CNN_CLASSES,
        title="Meta-LightGBM - Multiclass Confusion Matrix (OOF)",
        out_path=OUT_DIR / "confusion_multiclass.png",
    )

    # ── Feature importance plots ──────────────────────────────────────────────
    save_feature_importance(bin_model, bin_names, "binary",
                            OUT_DIR / "feature_importance_binary.png")
    save_feature_importance(mul_model, mul_names, "multiclass",
                            OUT_DIR / "feature_importance_multiclass.png")

    # ── Save feature names for later inference ────────────────────────────────
    json.dump({"binary": bin_names, "multiclass": mul_names},
              open(OUT_DIR / "meta_feature_names.json", "w"), indent=2)

    # ── Save per-fold metrics CSV ─────────────────────────────────────────────
    records = []
    for row in bin_folds:
        records.append({"task": "binary",     **row})
    records.append({"task": "binary",     **bin_oof})
    for row in mul_folds:
        records.append({"task": "multiclass", **row})
    records.append({"task": "multiclass", **mul_oof})
    pd.DataFrame(records).to_csv(OUT_DIR / "meta_results.csv", index=False)
    print(f"Saved {OUT_DIR / 'meta_results.csv'}")

    # ── Comparison tables: meta vs. individual base models ───────────────────
    # Load base model OOF metrics from existing metrics_summary.csv
    summary_csv = BASE / "models" / "ensemble_eval" / "metrics_summary.csv"
    if summary_csv.exists():
        base_df = pd.read_csv(summary_csv)
        bin_rows = []
        mul_rows = []
        for _, r in base_df.iterrows():
            if r["model"] == "Meta-LightGBM":
                continue  # skip stale row - we append a fresh OOF one below
            bin_rows.append({"model": r["model"],
                             "acc": r["bin_acc"], "f1": r["bin_f1"],
                             "prec": r["bin_prec"], "rec": r["bin_rec"],
                             "mcc": r["bin_mcc"], "auc": r["bin_auc"]})
            mul_rows.append({"model": r["model"],
                             "acc": r["mul_acc"], "f1": r["mul_f1"],
                             "prec": r["mul_prec"], "rec": r["mul_rec"],
                             "mcc": r["mul_mcc"], "auc": r["mul_auc"]})
        # Append meta-learner OOF result
        bin_rows.append({"model": "Meta-LightGBM",
                         **{k: bin_oof[k] for k in ["acc","f1","prec","rec","mcc","auc"]}})
        mul_rows.append({"model": "Meta-LightGBM",
                         **{k: mul_oof[k] for k in ["acc","f1","prec","rec","mcc","auc"]}})

        save_comparison_table(bin_rows, OUT_DIR / "metrics_table_binary.png",
                              "Binary Classification - Base Models vs. Meta-LightGBM")
        save_comparison_table(mul_rows, OUT_DIR / "metrics_table_multiclass.png",
                              "Multiclass Classification - Base Models vs. Meta-LightGBM")
    else:
        print("[!] metrics_summary.csv not found - skipping comparison tables.")


if __name__ == "__main__":
    main()
