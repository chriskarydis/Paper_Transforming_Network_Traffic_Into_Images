#!/usr/bin/env python3
"""Builds Combined_Greyscale and Combined_RGB image pools by symlinking the
packet and flow images for each encoding into a shared directory tree,
prefixing filenames to avoid collisions (packet and flow images can share
the same stem-based filename)."""
from pathlib import Path

EXP_DIR = Path(__file__).parent

PAIRS = {
    "images_combined_grey": (EXP_DIR / "images_packet", EXP_DIR / "images_flow_grey"),
    "images_combined_rgb": (EXP_DIR / "images_packet_rgb", EXP_DIR / "images_flow"),
}


def link_tree(src_root: Path, dst_root: Path, prefix: str):
    n = 0
    for split_dir in src_root.iterdir():
        if not split_dir.is_dir():
            continue
        for cls_dir in split_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            dst_dir = dst_root / split_dir.name / cls_dir.name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for img in cls_dir.glob("*.png"):
                link = dst_dir / f"{prefix}_{img.name}"
                if not link.exists():
                    link.symlink_to(img.resolve())
                n += 1
    return n


def main():
    for out_name, (pkt_root, flow_root) in PAIRS.items():
        out_root = EXP_DIR / out_name
        n_pkt = link_tree(pkt_root, out_root, "pkt")
        n_flow = link_tree(flow_root, out_root, "flow")
        print(f"{out_name}: {n_pkt} packet links + {n_flow} flow links -> {out_root}")


if __name__ == "__main__":
    main()
