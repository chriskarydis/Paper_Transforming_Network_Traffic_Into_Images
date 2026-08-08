#!/usr/bin/env python3
"""Builds stacked meta-features (concatenated class-probability vectors from
all 6 CNN configs + calibrated LightGBM) for the val and test splits, in
manifest row order, for training/evaluating the Meta-LightGBM stacker under
the unified split."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).parent

CNN_MODELS = [
    "Packet_Greyscale", "Packet_RGB",
    "Flow_Greyscale", "Flow_RGB",
    "Combined_Greyscale", "Combined_RGB",
]


def load_lgb(split, stems_ref):
    df = pd.read_csv(EXP_DIR / "features.csv")
    split_df = df[df["split"] == split].reset_index(drop=True)
    stems = split_df["pcap"].apply(lambda p: Path(p).stem).values
    assert np.array_equal(stems, stems_ref), f"LightGBM {split} order mismatch!"
    npz = np.load(EXP_DIR / "lightgbm_model" / f"{split}_predictions.npz")
    labels = npz["labels"] if "labels" in npz else npz["bin_labels"]
    bin_proba = npz["bin_proba"]
    if split == "test" and "proba_calibrated" in npz:
        mul_proba = npz["proba_calibrated"]
    else:
        # val_predictions.npz only stores raw proba, calibrate the same way
        thresh = np.load(EXP_DIR / "lightgbm_model" / "class_thresholds.npy")
        raw = npz["proba"]
        calibrated = raw * thresh
        mul_proba = calibrated / calibrated.sum(axis=1, keepdims=True)
    return mul_proba, bin_proba, labels


def main():
    for split in ("val", "test"):
        mul_blocks, bin_blocks = [], []
        stems_ref = None
        labels_ref = None
        for name in CNN_MODELS:
            npz = np.load(EXP_DIR / "models" / name / f"{split}_predictions_aligned.npz",
                          allow_pickle=True)
            if stems_ref is None:
                stems_ref = npz["stems"]
                labels_ref = npz["labels"]
            else:
                assert np.array_equal(npz["stems"], stems_ref), f"{name} {split} stem mismatch!"
                assert np.array_equal(npz["labels"], labels_ref), f"{name} {split} label mismatch!"
            mul_blocks.append(npz["mul_probs"])
            bin_blocks.append(npz["bin_probs"])

        lgb_mul, lgb_bin, lgb_labels = load_lgb(split, stems_ref)
        assert np.array_equal(lgb_labels, labels_ref), f"LightGBM {split} label mismatch!"
        mul_blocks.append(lgb_mul)
        bin_blocks.append(lgb_bin)

        X_mul = np.hstack(mul_blocks)
        X_bin = np.hstack(bin_blocks)
        np.savez(EXP_DIR / f"meta_features_{split}.npz",
                  X_mul=X_mul, X_bin=X_bin, labels=labels_ref, stems=stems_ref)
        print(f"{split}: X_mul {X_mul.shape}  X_bin {X_bin.shape}  n={len(labels_ref)}")

    json.dump({"models": CNN_MODELS + ["LightGBM"]},
              open(EXP_DIR / "meta_features_models.json", "w"), indent=2)


if __name__ == "__main__":
    main()
