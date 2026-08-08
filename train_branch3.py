#!/usr/bin/env python3
"""
train_branch3.py
Branch 3: Per-flow statistical feature extraction + LightGBM classifier.
Extracts pcap2csv-inspired features from each flow pcap,
trains binary + multi-class LightGBM models, evaluates and saves them.
"""

import os
import sys
# Prevent LightGBM's OpenMP from deadlocking under WSL2 + joblib fork
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from scipy.optimize import differential_evolution

import dpkt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (confusion_matrix, f1_score, accuracy_score)
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import joblib
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BASE      = Path.home() / "thesis_heifip"
FLOW_DIR  = BASE / "data" / "flows"
OUT_DIR   = BASE / "models" / "branch3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED      = 42
CAP_FLOWS = 1000  # max flows per class to keep memory manageable
random.seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# Feature extraction (per flow pcap)
# Produces one feature vector per pcap file.
# Features mirror pcap2csv column set.
# ─────────────────────────────────────────────
def extract_features_from_flow(pcap_path: str) -> dict | None:
    """
    Reads a single flow pcap and returns a dict of 39 statistical features.
    Returns None if the pcap is unreadable or has no valid packets.
    """
    try:
        f = open(pcap_path, "rb")
        pcap = dpkt.pcap.Reader(f)
    except Exception:
        return None

    sizes        = []
    timestamps   = []
    iats         = []
    header_lens  = []
    ttls         = []
    proto_types  = []
    dst_ports    = []
    src_ports    = []

    fin_count = syn_count = rst_count = 0
    psh_count = ack_count = ece_count = cwr_count = 0

    http = https = dns = telnet = smtp = ssh = irc = 0
    tcp  = udp   = dhcp = arp   = icmp = igmp = ipv = 0
    ftp  = netbios = 0

    last_ts = None
    count   = 0

    try:
        for ts, buf in pcap:
            count += 1
            sizes.append(len(buf))
            timestamps.append(ts)

            if last_ts is not None:
                iats.append(ts - last_ts)
            last_ts = ts

            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except Exception:
                continue

            if eth.type == dpkt.ethernet.ETH_TYPE_ARP:
                arp = 1
                continue

            if eth.type != dpkt.ethernet.ETH_TYPE_IP:
                continue

            ipv = 1
            ip  = eth.data
            if not isinstance(ip, dpkt.ip.IP):
                continue

            ttls.append(ip.ttl)
            proto_types.append(ip.p)

            if isinstance(ip.data, dpkt.icmp.ICMP):
                icmp = 1
                continue

            if isinstance(ip.data, dpkt.igmp.IGMP):
                igmp = 1
                continue

            if isinstance(ip.data, dpkt.udp.UDP):
                udp_pkt = ip.data
                udp = 1
                header_lens.append(8)
                sport, dport = udp_pkt.sport, udp_pkt.dport
                dst_ports.append(dport)
                src_ports.append(sport)
                if dport == 53 or sport == 53:   dns     = 1
                if dport == 67 or dport == 68:   dhcp    = 1
                if dport in (137, 138) or sport in (137, 138): netbios = 1

            elif isinstance(ip.data, dpkt.tcp.TCP):
                tcp_pkt = ip.data
                tcp = 1
                header_lens.append(tcp_pkt.off * 4)
                sport, dport = tcp_pkt.sport, tcp_pkt.dport
                dst_ports.append(dport)
                src_ports.append(sport)
                flags = tcp_pkt.flags
                if flags & dpkt.tcp.TH_FIN: fin_count += 1
                if flags & dpkt.tcp.TH_SYN: syn_count += 1
                if flags & dpkt.tcp.TH_RST: rst_count += 1
                if flags & dpkt.tcp.TH_PUSH:psh_count += 1
                if flags & dpkt.tcp.TH_ACK: ack_count += 1
                if flags & 0x40:            ece_count += 1
                if flags & 0x80:            cwr_count += 1
                if dport == 80   or sport == 80:    http    = 1
                if dport == 443  or sport == 443:   https   = 1
                if dport == 22   or sport == 22:    ssh     = 1
                if dport == 23   or sport == 23:    telnet  = 1
                if dport == 25   or sport == 25:    smtp    = 1
                if dport in (194, 6667, 6668, 6697) or \
                   sport in (194, 6667, 6668, 6697): irc    = 1
                if dport in (20, 21) or sport in (20, 21): ftp = 1
                if dport in (139, 445) or sport in (139, 445): netbios = 1
            else:
                continue

    except Exception:
        pass
    finally:
        f.close()

    if not sizes:
        return None

    sizes_arr = np.array(sizes)
    iats_arr  = np.array(iats) if iats else np.array([0.0])
    duration  = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0

    sq25, sq50, sq75   = np.percentile(sizes_arr, [25, 50, 75])
    iq25, iq50, iq75   = np.percentile(iats_arr,  [25, 50, 75])
    # dst_port_mode / src_port_mode are the strongest discriminators between attack types
    _mode = lambda lst: max(set(lst), key=lst.count) if lst else 0

    return {
        "pkt_count":          count,
        "duration":           duration,
        "rate":               count / (duration + 1e-9),
        "bytes_per_sec":      sizes_arr.sum() / (duration + 1e-9),
        "tot_size":           sizes_arr.sum(),
        "min_size":           sizes_arr.min(),
        "max_size":           sizes_arr.max(),
        "avg_size":           sizes_arr.mean(),
        "std_size":           sizes_arr.std(),
        "var_size":           sizes_arr.var(),
        "size_range":         sizes_arr.max() - sizes_arr.min(),
        "size_q25":           float(sq25),
        "size_q50":           float(sq50),
        "size_q75":           float(sq75),
        "iat_mean":           iats_arr.mean(),
        "iat_std":            iats_arr.std(),
        "iat_min":            iats_arr.min(),
        "iat_max":            iats_arr.max(),
        "iat_var":            iats_arr.var(),
        "iat_q25":            float(iq25),
        "iat_q50":            float(iq50),
        "iat_q75":            float(iq75),
        "header_len_mean":    float(np.mean(header_lens)) if header_lens else 0,
        "ttl_mean":           float(np.mean(ttls))        if ttls        else 0,
        "ttl_std":            float(np.std(ttls))         if ttls        else 0,
        "proto_type_mode":    int(_mode(proto_types)),
        "dst_port_mode":      int(_mode(dst_ports)),
        "src_port_mode":      int(_mode(src_ports)),
        "n_unique_dst_ports": len(set(dst_ports)),
        "n_unique_src_ports": len(set(src_ports)),
        "bytes_per_pkt":      sizes_arr.sum() / (count + 1e-9),
        "psh_ack_ratio":      psh_count / (ack_count + 1e-9),
        "ack_rate":           ack_count / (count + 1e-9),
        "fin_count":          fin_count,
        "syn_count":          syn_count,
        "rst_count":          rst_count,
        "psh_count":          psh_count,
        "ack_count":          ack_count,
        "ece_count":          ece_count,
        "cwr_count":          cwr_count,
        "syn_rate":           syn_count / (count + 1e-9),
        "fin_rate":           fin_count / (count + 1e-9),
        "rst_rate":           rst_count / (count + 1e-9),
        "f_http":             http,
        "f_https":            https,
        "f_dns":              dns,
        "f_telnet":           telnet,
        "f_smtp":             smtp,
        "f_ssh":              ssh,
        "f_irc":              irc,
        "f_ftp":              ftp,
        "f_netbios":          netbios,
        "f_tcp":              tcp,
        "f_udp":              udp,
        "f_dhcp":             dhcp,
        "f_arp":              arp,
        "f_icmp":             icmp,
        "f_igmp":             igmp,
        "f_ipv":              ipv,
    }

# ─────────────────────────────────────────────
# Dataset collection
# ─────────────────────────────────────────────
def collect_dataset() -> pd.DataFrame:
    records = []

    for category_dir in sorted(FLOW_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        is_malicious = "malicious" in category_dir.name

        # Accumulate all flow pcaps per class before capping and processing
        # (merges all benign subdirs like cic2017_benign into normal_browsing)
        class_pcaps = defaultdict(list)
        for class_dir in sorted(category_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name if is_malicious else "normal_browsing"
            class_pcaps[class_name].extend(class_dir.glob("*.pcap"))

        binary_label = 1 if is_malicious else 0
        for class_name, flow_pcaps in sorted(class_pcaps.items()):
            if not flow_pcaps:
                continue
            if len(flow_pcaps) > CAP_FLOWS:
                flow_pcaps = random.sample(flow_pcaps, CAP_FLOWS)

            print(f"  [{class_name}] {len(flow_pcaps)} flows")

            for fp in tqdm(flow_pcaps, desc=f"    {class_name}", leave=False):
                feats = extract_features_from_flow(str(fp))
                if feats is None:
                    continue
                feats["class_name"]   = class_name
                feats["binary_label"] = binary_label
                records.append(feats)

    df = pd.DataFrame(records)
    print(f"\n[✓] Total samples collected: {len(df)}")
    print(f"    Classes: {sorted(df['class_name'].unique())}")
    return df

# ─────────────────────────────────────────────
# Train + evaluate
# ─────────────────────────────────────────────
def train_and_evaluate(df: pd.DataFrame):
    feature_cols = [c for c in df.columns
                    if c not in ("class_name", "binary_label")]

    X = df[feature_cols].values
    y_bin = df["binary_label"].values

    le = LabelEncoder()
    y_mul = le.fit_transform(df["class_name"].values)
    joblib.dump(le, OUT_DIR / "label_encoder.pkl")

    with open(OUT_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    K = 5
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)

    print(f"\n[Binary] {K}-fold cross-validation...")
    bin_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    lgb.LGBMClassifier(
            n_estimators=400,
            num_leaves=63,
            learning_rate=0.05,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            min_child_samples=10,
            reg_alpha=0.05,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=1,
            verbose=-1,
        ))
    ])

    bin_proba_cv = cross_val_predict(bin_pipeline, X, y_bin, cv=skf,
                                     method="predict_proba", n_jobs=1)
    bin_fold_f1  = np.array([
        f1_score(y_bin[vi], bin_proba_cv[vi].argmax(axis=1), average="macro")
        for _, vi in skf.split(X, y_bin)
    ])
    bin_fold_acc = np.array([
        accuracy_score(y_bin[vi], bin_proba_cv[vi].argmax(axis=1))
        for _, vi in skf.split(X, y_bin)
    ])
    bin_acc_mean, bin_acc_std = bin_fold_acc.mean(), bin_fold_acc.std()
    bin_f1_mean,  bin_f1_std  = bin_fold_f1.mean(),  bin_fold_f1.std()

    print(f"  Accuracy : {bin_acc_mean:.4f} ± {bin_acc_std:.4f}")
    print(f"  Macro F1 : {bin_f1_mean:.4f} ± {bin_f1_std:.4f}")

    print("[Binary] Refitting on full dataset...")
    bin_pipeline.fit(X, y_bin)
    joblib.dump(bin_pipeline, OUT_DIR / "binary_model.pkl")

    print(f"\n[Multi] {K}-fold cross-validation + OOF probabilities...")
    # SMOTE k_neighbors must be < smallest training-fold class size.
    # Smallest class (zeus=298) after augmentation → k=2 still safe in 80% folds.
    smote = SMOTE(random_state=SEED, k_neighbors=2)

    mul_pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  smote),
        ("clf",    lgb.LGBMClassifier(
            objective="multiclass",
            n_estimators=1000,
            num_leaves=127,
            learning_rate=0.02,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            min_child_samples=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=1,
            verbose=-1,
        ))
    ])

    # Single cross_val_predict pass gives OOF probas. Derive fold metrics from splits.
    ym_proba_cv = cross_val_predict(mul_pipeline, X, y_mul, cv=skf,
                                    method="predict_proba", n_jobs=1)
    mul_fold_f1  = np.array([
        f1_score(y_mul[vi], ym_proba_cv[vi].argmax(axis=1), average="macro")
        for _, vi in skf.split(X, y_mul)
    ])
    mul_fold_acc = np.array([
        accuracy_score(y_mul[vi], ym_proba_cv[vi].argmax(axis=1))
        for _, vi in skf.split(X, y_mul)
    ])
    mul_acc_mean, mul_acc_std = mul_fold_acc.mean(), mul_fold_acc.std()
    mul_f1_mean,  mul_f1_std  = mul_fold_f1.mean(),  mul_fold_f1.std()

    print(f"  Accuracy : {mul_acc_mean:.4f} ± {mul_acc_std:.4f}")
    print(f"  Macro F1 : {mul_f1_mean:.4f} ± {mul_f1_std:.4f}")

    print("[Multi] Refitting on full dataset...")
    mul_pipeline.fit(X, y_mul)
    clf_mul = mul_pipeline.named_steps["clf"]
    joblib.dump(mul_pipeline, OUT_DIR / "multi_model.pkl")

    print("[Multi] Calibrating per-class probability thresholds...")
    n_classes = len(le.classes_)

    def neg_macro_f1(thresholds):
        preds = (ym_proba_cv * thresholds).argmax(axis=1)
        return -f1_score(y_mul, preds, average="macro", zero_division=0)

    opt = differential_evolution(
        neg_macro_f1,
        bounds=[(0.1, 10.0)] * n_classes,
        seed=SEED, maxiter=300, tol=1e-4, polish=True,
    )
    best_thresholds = opt.x
    np.save(OUT_DIR / "class_thresholds.npy", best_thresholds)

    ym_pred_tuned = (ym_proba_cv * best_thresholds).argmax(axis=1)
    tuned_f1 = f1_score(y_mul, ym_pred_tuned, average="macro")
    print(f"  Macro F1 (OOF, raw)      : {mul_f1_mean:.4f}")
    print(f"  Macro F1 (OOF, calibrated): {tuned_f1:.4f}")

    # Use calibrated predictions for the confusion matrix
    cm = confusion_matrix(y_mul, ym_pred_tuned)
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title(f"Branch 3 LightGBM - Multi-class Confusion Matrix ({K}-fold CV)")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_multiclass.png", dpi=150)
    plt.close()

    # ── CV scores per fold plot ────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Branch 3 LightGBM - {K}-Fold Cross-Validation Scores")

    for ax, scores, title in [
        (axes[0], bin_fold_f1,             "Binary Macro F1"),
        (axes[1], mul_fold_f1,             "Multi-class Macro F1"),
    ]:
        ax.bar(range(1, K+1), scores, color="#2196F3")
        ax.axhline(scores.mean(), color="red", linestyle="--",
                   label=f"Mean: {scores.mean():.4f}")
        ax.set_xlabel("Fold")
        ax.set_ylabel("Macro F1")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "cv_scores.png", dpi=150)
    plt.close()

    # ── Feature importance ─────────────────────
    importances = clf_mul.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    feat_imp.head(20).plot(kind="bar")
    plt.title("Branch 3 LightGBM - Top 20 Feature Importances (Multi-class)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_importance.png", dpi=150)
    plt.close()

    # ── Save results ───────────────────────────
    results = {
        "binary": {
            "cv_accuracy_mean": round(bin_acc_mean, 4),
            "cv_accuracy_std":  round(bin_acc_std,  4),
            "cv_macro_f1_mean": round(bin_f1_mean,  4),
            "cv_macro_f1_std":  round(bin_f1_std,   4),
            "fold_f1_scores":   bin_fold_f1.tolist(),
        },
        "multiclass": {
            "cv_accuracy_mean":      round(mul_acc_mean, 4),
            "cv_accuracy_std":       round(mul_acc_std,  4),
            "cv_macro_f1_mean":      round(mul_f1_mean,  4),
            "cv_macro_f1_std":       round(mul_f1_std,   4),
            "fold_f1_scores":        mul_fold_f1.tolist(),
            "calibrated_macro_f1":   round(tuned_f1, 4),
            "class_thresholds":      best_thresholds.tolist(),
        },
        "k_folds":    K,
        "classes":    list(le.classes_),
        "n_features": len(feature_cols),
        "n_samples":  len(df),
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[✓] Models saved to {OUT_DIR}")
    print(f"\n  Binary  - Accuracy: {bin_acc_mean:.4f} ± {bin_acc_std:.4f} | "
          f"Macro F1: {bin_f1_mean:.4f} ± {bin_f1_std:.4f}")
    print(f"  Multi   - Accuracy: {mul_acc_mean:.4f} ± {mul_acc_std:.4f} | "
          f"Macro F1 (raw): {mul_f1_mean:.4f} ± {mul_f1_std:.4f} | "
          f"Macro F1 (calibrated): {tuned_f1:.4f}")

    return results

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  Branch 3: Flow Feature Extraction + LightGBM")
    print("="*60)

    print("\nCollecting dataset from flow pcaps...")
    df = collect_dataset()

    if df.empty:
        print("[!] No samples collected. Check FLOW_DIR path.")
        sys.exit(1)

    # Replace inf/nan from division edge cases
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    df.to_csv(OUT_DIR / "features_dataset.csv", index=False)
    print(f"[✓] Feature dataset saved to {OUT_DIR / 'features_dataset.csv'}")

    train_and_evaluate(df)

if __name__ == "__main__":
    main()