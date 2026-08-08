#!/usr/bin/env python3
"""
Runs inference for a trained CNN branch (Packet_Greyscale or Flow_RGB) over the
TEST split, strictly following manifest.csv's row order, keyed by flow stem.
This guarantees alignment with the other branches' test predictions, which is
not guaranteed by glob()-based directory iteration order.
"""
import csv
import sys
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

IMG_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXP_DIR = Path(__file__).parent
MANIFEST = EXP_DIR / "manifest.csv"

CNN_CLASSES = [
    "0day", "blackEnergy", "distcc_exec_backdoor",
    "hydra_ftp", "hydra_ssh", "java_rmi",
    "mirai", "netbios_ssn", "normal_browsing",
    "replayAttacks", "ruby_drb", "smtp",
    "tomcat", "unreallrcd", "vsftpd", "zeus",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CNN_CLASSES)}
NUM_CLASSES = len(CNN_CLASSES)

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


class DualHeadModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.binary_head = nn.Linear(in_features, 2)
        self.multi_head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        f = self.backbone(x)
        return self.binary_head(f), self.multi_head(f)


def load_images(paths):
    tensors = []
    for p in paths:
        try:
            tensors.append(TRANSFORM(Image.open(p).convert("RGB")))
        except Exception:
            continue
    return tensors


def main():
    ap = ArgumentParser()
    ap.add_argument("--mode", choices=["packet", "flow", "combined"], required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = DualHeadModel(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()

    rows = [r for r in csv.DictReader(open(MANIFEST)) if r["split"] == args.split]
    print(f"Predicting {len(rows)} flows ({args.mode}) from {args.images_root}")

    images_root = Path(args.images_root) / args.split
    bin_probs, mul_probs, labels, stems = [], [], [], []

    with torch.no_grad():
        for i, r in enumerate(rows):
            cls = r["class"]
            stem = Path(r["pcap"]).stem
            cls_dir = images_root / cls
            if args.mode == "flow":
                paths = [cls_dir / f"{stem}_0.png"]
                paths = [p for p in paths if p.exists()]
            elif args.mode == "combined":
                paths = sorted(cls_dir.glob(f"*_{stem}_*.png"))
            else:
                paths = sorted(cls_dir.glob(f"{stem}_*.png"))

            if not paths:
                bin_probs.append([0.5, 0.5])
                mul_probs.append(np.ones(NUM_CLASSES) / NUM_CLASSES)
            else:
                tensors = load_images(paths)
                if not tensors:
                    bin_probs.append([0.5, 0.5])
                    mul_probs.append(np.ones(NUM_CLASSES) / NUM_CLASSES)
                else:
                    batch = torch.stack(tensors).to(DEVICE)
                    bin_out, mul_out = model(batch)
                    bp = torch.softmax(bin_out, 1).mean(0).cpu().numpy()
                    mp = torch.softmax(mul_out, 1).mean(0).cpu().numpy()
                    bin_probs.append(bp)
                    mul_probs.append(mp)

            labels.append(CLASS_TO_IDX[cls])
            stems.append(stem)

            if (i + 1) % 200 == 0:
                print(f"  [{i+1}/{len(rows)}]")

    np.savez(args.out,
             bin_probs=np.array(bin_probs), mul_probs=np.array(mul_probs),
             labels=np.array(labels), stems=np.array(stems))
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
