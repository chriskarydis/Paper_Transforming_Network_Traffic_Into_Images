#!/usr/bin/env python3
import sys
import csv
import time
from pathlib import Path

sys.path.insert(0, str(Path.home() / "thesis_heifip"))
from train_branch3 import extract_features_from_flow

MANIFEST = Path(__file__).parent / "manifest.csv"
OUT_CSV = Path(__file__).parent / "features.csv"

def main():
    rows = list(csv.DictReader(open(MANIFEST)))
    print(f"Total flows: {len(rows)}")

    t0 = time.time()
    records = []
    feature_cols = None
    skipped = 0

    for i, r in enumerate(rows):
        feats = extract_features_from_flow(r["pcap"])
        if feats is None:
            skipped += 1
            continue
        if feature_cols is None:
            feature_cols = list(feats.keys())
        feats["class"] = r["class"]
        feats["binary"] = r["binary"]
        feats["split"] = r["split"]
        feats["pcap"] = r["pcap"]
        records.append(feats)

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(rows)}] elapsed={elapsed:.1f}s  skipped={skipped}")

    print(f"\nExtracted features for {len(records)} flows (skipped {skipped})")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    with open(OUT_CSV, "w", newline="") as f:
        fieldnames = feature_cols + ["class", "binary", "split", "pcap"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)
    print(f"Saved to {OUT_CSV}")

if __name__ == "__main__":
    main()
