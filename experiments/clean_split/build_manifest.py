#!/usr/bin/env python3
import random
import csv
from pathlib import Path
from collections import defaultdict
import dpkt
from sklearn.model_selection import train_test_split

SEED = 42
CAP_PER_CLASS = 500
MIN_PKTS = 1
EXCLUDE_PREFIXES = ("ether_", "dot3_")

BASE = Path.home() / "thesis_heifip"
FLOWS_DIR = BASE / "data" / "flows"

# class -> list of source dirs to pool flows from
CLASS_SOURCES = {
    "0day":                  [FLOWS_DIR / "malicious_pcaps" / "0day"],
    "blackEnergy":            [FLOWS_DIR / "malicious_pcaps" / "blackEnergy"],
    "distcc_exec_backdoor":   [FLOWS_DIR / "malicious_pcaps" / "distcc_exec_backdoor"],
    "hydra_ftp":              [FLOWS_DIR / "malicious_pcaps" / "hydra_ftp"],
    "hydra_ssh":              [FLOWS_DIR / "malicious_pcaps" / "hydra_ssh"],
    "java_rmi":               [FLOWS_DIR / "malicious_pcaps" / "java_rmi"],
    "mirai":                  [FLOWS_DIR / "malicious_pcaps" / "mirai"],
    "netbios_ssn":            [FLOWS_DIR / "malicious_pcaps" / "netbios_ssn"],
    "normal_browsing":        [FLOWS_DIR / "benign_pcaps" / "normal_browsing",
                                FLOWS_DIR / "benign_pcaps" / "cic2017_benign"],
    "replayAttacks":          [FLOWS_DIR / "malicious_pcaps" / "replayAttacks"],
    "ruby_drb":                [FLOWS_DIR / "malicious_pcaps" / "ruby_drb"],
    "smtp":                   [FLOWS_DIR / "malicious_pcaps" / "smtp"],
    "tomcat":                 [FLOWS_DIR / "malicious_pcaps" / "tomcat"],
    "unreallrcd":             [FLOWS_DIR / "malicious_pcaps" / "unreallrcd"],
    "vsftpd":                 [FLOWS_DIR / "malicious_pcaps" / "vsftpd"],
    "zeus":                   [FLOWS_DIR / "malicious_pcaps" / "zeus"],
}
BENIGN_CLASS = "normal_browsing"

random.seed(SEED)

def count_packets(pcap_path):
    try:
        with open(pcap_path, "rb") as f:
            r = dpkt.pcap.Reader(f)
            n = 0
            for _ in r:
                n += 1
                if n >= MIN_PKTS:
                    return n
            return n
    except Exception:
        return 0

def main():
    all_rows = []
    for cls, src_dirs in CLASS_SOURCES.items():
        candidates = []
        for d in src_dirs:
            if not d.exists():
                continue
            candidates.extend(
                p for p in d.iterdir()
                if p.suffix == ".pcap" and not p.name.startswith(EXCLUDE_PREFIXES)
            )
        print(f"{cls}: {len(candidates)} raw flow pcaps found across {len(src_dirs)} source dir(s)")

        random.shuffle(candidates)
        kept = []
        for p in candidates:
            if len(kept) >= CAP_PER_CLASS:
                break
            n = count_packets(p)
            if n >= MIN_PKTS:
                kept.append(p)
        print(f"  -> kept {len(kept)} flows (>= {MIN_PKTS} pkts, cap {CAP_PER_CLASS})")

        bin_lbl = 0 if cls == BENIGN_CLASS else 1
        for p in kept:
            all_rows.append({"pcap": str(p), "class": cls, "binary": bin_lbl})

    print(f"\nTotal flows collected: {len(all_rows)}")

    # Stratified 70/15/15 split
    train_rows, temp_rows = train_test_split(
        all_rows, test_size=0.30, random_state=SEED,
        stratify=[r["class"] for r in all_rows])
    val_rows, test_rows = train_test_split(
        temp_rows, test_size=0.50, random_state=SEED,
        stratify=[r["class"] for r in temp_rows])

    for r in train_rows: r["split"] = "train"
    for r in val_rows:   r["split"] = "val"
    for r in test_rows:  r["split"] = "test"

    final = train_rows + val_rows + test_rows
    print(f"Train: {len(train_rows)}  Val: {len(val_rows)}  Test: {len(test_rows)}")

    out_csv = Path(__file__).parent / "manifest.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pcap", "class", "binary", "split"])
        writer.writeheader()
        for r in final:
            writer.writerow(r)
    print(f"Saved manifest to {out_csv}")

    # Per-class per-split breakdown
    breakdown = defaultdict(lambda: defaultdict(int))
    for r in final:
        breakdown[r["class"]][r["split"]] += 1
    print(f"\n{'class':<24}{'train':>8}{'val':>8}{'test':>8}")
    for cls in CLASS_SOURCES:
        b = breakdown[cls]
        print(f"{cls:<24}{b['train']:>8}{b['val']:>8}{b['test']:>8}")

if __name__ == "__main__":
    main()
