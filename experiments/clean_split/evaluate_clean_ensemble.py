#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (f1_score, accuracy_score, precision_score, recall_score,
                              roc_auc_score, matthews_corrcoef)

EXP_DIR = Path(__file__).parent

CNN_CLASSES = [
    "0day", "blackEnergy", "distcc_exec_backdoor",
    "hydra_ftp", "hydra_ssh", "java_rmi",
    "mirai", "netbios_ssn", "normal_browsing",
    "replayAttacks", "ruby_drb", "smtp",
    "tomcat", "unreallrcd", "vsftpd", "zeus",
]


def load_lgb_test(stems_ref):
    df = pd.read_csv(EXP_DIR / "features.csv")
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    stems = test_df["pcap"].apply(lambda p: Path(p).stem).values
    assert np.array_equal(stems, stems_ref), "LightGBM test order mismatch with reference!"

    lgb_npz = np.load(EXP_DIR / "lightgbm_model" / "test_predictions.npz")
    return lgb_npz["proba"], lgb_npz["proba_calibrated"], lgb_npz["labels"]


def metrics_multiclass(y_true, P, name):
    y_pred = P.argmax(1)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    try:
        present = sorted(set(y_true.tolist()))
        auc = roc_auc_score(y_true, P, multi_class="ovr", average="macro",
                             labels=list(range(len(CNN_CLASSES))))
    except Exception:
        auc = float("nan")
    print(f"  {name:20s} acc={acc:.4f} f1={f1:.4f} prec={prec:.4f} rec={rec:.4f} mcc={mcc:.4f} auc={auc:.4f}")
    return {"acc": acc, "f1": f1, "prec": prec, "rec": rec, "mcc": mcc, "auc": auc}


def bin_proba_from_mul(P):
    benign_idx = CNN_CLASSES.index("normal_browsing")
    p_benign = P[:, benign_idx]
    p_mal = 1 - p_benign
    return np.stack([p_benign, p_mal], axis=1)


def metrics_binary(y_true_bin, P_bin, name):
    y_pred = P_bin.argmax(1)
    acc = accuracy_score(y_true_bin, y_pred)
    f1 = f1_score(y_true_bin, y_pred, average="macro")
    mcc = matthews_corrcoef(y_true_bin, y_pred)
    try:
        auc = roc_auc_score(y_true_bin, P_bin[:, 1])
    except Exception:
        auc = float("nan")
    print(f"  {name:20s} acc={acc:.4f} f1={f1:.4f} mcc={mcc:.4f} auc={auc:.4f}")
    return {"acc": acc, "f1": f1, "mcc": mcc, "auc": auc}


def main():
    pkt = np.load(EXP_DIR / "models" / "Packet_Greyscale" / "test_predictions_aligned.npz", allow_pickle=True)
    flw = np.load(EXP_DIR / "models" / "Flow_RGB" / "test_predictions_aligned.npz", allow_pickle=True)

    assert np.array_equal(pkt["stems"], flw["stems"]), "Packet/Flow test order mismatch!"
    assert np.array_equal(pkt["labels"], flw["labels"]), "Packet/Flow test label mismatch!"

    P_lgb_raw, P_lgb_cal, lgb_labels = load_lgb_test(pkt["stems"])
    assert np.array_equal(lgb_labels, pkt["labels"]), "LightGBM test label mismatch!"

    y_true = pkt["labels"]
    P_pkt = pkt["mul_probs"]
    P_flw = flw["mul_probs"]
    P_lgb = P_lgb_cal  # calibrated, matches original methodology (calibrated model used in ensemble)

    pkt_summary = json.load(open(EXP_DIR / "models" / "Packet_Greyscale" / "summary.json"))
    flw_summary = json.load(open(EXP_DIR / "models" / "Flow_RGB" / "summary.json"))
    lgb_results = json.load(open(EXP_DIR / "lightgbm_model" / "results.json"))

    w1 = pkt_summary["best_val_mul_f1"]
    w2 = flw_summary["best_val_mul_f1"]
    w3 = lgb_results["multiclass"]["val_f1_calibrated"]
    print(f"Ensemble weights (held-out val macro F1): w_packet={w1:.4f} w_flow={w2:.4f} w_lgbm={w3:.4f}\n")

    P_ens = (w1 * P_pkt + w2 * P_flw + w3 * P_lgb) / (w1 + w2 + w3)

    print(f"=== Multi-class TEST metrics (clean, leak-free split, n={len(y_true)}) ===")
    r_pkt = metrics_multiclass(y_true, P_pkt, "Packet_Greyscale")
    r_flw = metrics_multiclass(y_true, P_flw, "Flow_RGB")
    r_lgb = metrics_multiclass(y_true, P_lgb, "LightGBM (calibrated)")
    r_ens = metrics_multiclass(y_true, P_ens, "WSV Ensemble")

    bin_labels = (np.array([CNN_CLASSES[i] for i in y_true]) != "normal_browsing").astype(int)
    print(f"\n=== Binary TEST metrics ===")
    rb_pkt = metrics_binary(bin_labels, bin_proba_from_mul(P_pkt), "Packet_Greyscale")
    rb_flw = metrics_binary(bin_labels, bin_proba_from_mul(P_flw), "Flow_RGB")
    rb_lgb = metrics_binary(bin_labels, bin_proba_from_mul(P_lgb), "LightGBM (calibrated)")
    rb_ens = metrics_binary(bin_labels, bin_proba_from_mul(P_ens), "WSV Ensemble")

    out = {
        "weights": {"w_packet": w1, "w_flow": w2, "w_lgbm": w3},
        "n_test": int(len(y_true)),
        "multiclass": {"packet": r_pkt, "flow": r_flw, "lgbm": r_lgb, "ensemble": r_ens},
        "binary": {"packet": rb_pkt, "flow": rb_flw, "lgbm": rb_lgb, "ensemble": rb_ens},
    }
    with open(EXP_DIR / "final_clean_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {EXP_DIR / 'final_clean_results.json'}")


if __name__ == "__main__":
    main()
