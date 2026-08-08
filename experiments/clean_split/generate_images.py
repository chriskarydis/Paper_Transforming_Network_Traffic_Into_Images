#!/usr/bin/env python3
import csv
import subprocess
import time
from pathlib import Path

BASE = Path.home() / "thesis_heifip"
HEIFIP = BASE / "tools" / "heiFIP" / "heiFIP" / "build" / "heiFIP"
MANIFEST = Path(__file__).parent / "manifest.csv"
OUT_PACKET = Path(__file__).parent / "images_packet"
OUT_FLOW = Path(__file__).parent / "images_flow"
TIMEOUT = 30

def main():
    rows = list(csv.DictReader(open(MANIFEST)))
    print(f"Total flows to process: {len(rows)}")

    t0 = time.time()
    n_pkt_imgs = 0
    n_flow_imgs = 0
    errors = 0

    for i, r in enumerate(rows):
        pcap = Path(r["pcap"])
        cls = r["class"]
        split = r["split"]
        stem = pcap.stem

        pkt_dir = OUT_PACKET / split / cls
        flow_dir = OUT_FLOW / split / cls
        pkt_dir.mkdir(parents=True, exist_ok=True)
        flow_dir.mkdir(parents=True, exist_ok=True)

        try:
            subprocess.run(
                [str(HEIFIP), "-i", str(pcap), "-o", str(pkt_dir),
                 "-m", "PacketImage", "--min-pkts", "1", "--name", stem],
                capture_output=True, timeout=TIMEOUT)
            subprocess.run(
                [str(HEIFIP), "-i", str(pcap), "-o", str(flow_dir),
                 "-m", "FlowImage", "--rgb", "--min-pkts", "1", "--name", stem],
                capture_output=True, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            errors += 1
            continue

        if (i + 1) % 250 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(rows) - i - 1)
            print(f"  [{i+1}/{len(rows)}] elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min  errors={errors}")

    n_pkt_imgs = sum(1 for _ in OUT_PACKET.rglob("*.png"))
    n_flow_imgs = sum(1 for _ in OUT_FLOW.rglob("*.png"))
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
    print(f"Packet images: {n_pkt_imgs}")
    print(f"Flow images:   {n_flow_imgs}")
    print(f"Errors/timeouts: {errors}")

if __name__ == "__main__":
    main()
