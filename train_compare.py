#!/usr/bin/env python3
"""
train_compare.py
Runs six experiments comparing greyscale vs RGB for:
1. PacketImage greyscale
2. PacketImage RGB
3. FlowImage greyscale
4. FlowImage RGB
5. Combined greyscale
6. Combined RGB
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BASE       = Path.home() / "thesis_heifip"
PACKET_DIR = BASE / "data" / "images_packet"
FLOW_DIR   = BASE / "data" / "images_flow"
OUTPUT_DIR = BASE / "models" / "comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE   = 32
BATCH_SIZE = 128
EPOCHS     = 15
LR         = 1e-3
SEED       = 42
CAP_PACKET = 5000
CAP_FLOW   = 500

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─────────────────────────────────────────────
# Dataset Builder
# base_dir: images_packet/ or images_flow/
# mode:     "greyscale" or "rgb"
# ─────────────────────────────────────────────
def collect_images(base_dir, mode, cap):
    """
    Expects structure:
      base_dir/
        greyscale/
          benign_pcaps/   <class_dir>/  *.png
          malicious_pcaps/<class_dir>/  *.png
        rgb/
          benign_pcaps/   <class_dir>/  *.png
          malicious_pcaps/<class_dir>/  *.png
    """
    samples = []
    mode_dir = Path(base_dir) / mode
    if not mode_dir.exists():
        print(f"  [!] Directory not found: {mode_dir}")
        return samples

    for category_dir in sorted(mode_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        is_malicious = "malicious" in category_dir.name
        for class_dir in sorted(category_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            images = list(class_dir.glob("*.png"))
            if len(images) == 0:
                continue
            if len(images) > cap:
                images = random.sample(images, cap)
            binary_label = 1 if is_malicious else 0
            for img_path in images:
                samples.append((str(img_path), binary_label, class_name))
    return samples

# ─────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────
class TrafficImageDataset(Dataset):
    def __init__(self, samples, class_to_idx, transform=None):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, binary_label, class_name = self.samples[idx]
        multi_label = self.class_to_idx[class_name]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), 0)
        if self.transform:
            img = self.transform(img)
        return img, binary_label, multi_label

# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
class DualHeadModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone    = backbone
        self.binary_head = nn.Linear(in_features, 2)
        self.multi_head  = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.binary_head(features), self.multi_head(features)

# ─────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = binary_correct = multi_correct = total = 0
    for imgs, bin_lbl, mul_lbl in loader:
        imgs, bin_lbl, mul_lbl = imgs.to(device), bin_lbl.to(device), mul_lbl.to(device)
        optimizer.zero_grad()
        bin_out, mul_out = model(imgs)
        loss = criterion(bin_out, bin_lbl) + criterion(mul_out, mul_lbl)
        loss.backward()
        optimizer.step()
        total_loss     += loss.item()
        binary_correct += (bin_out.argmax(1) == bin_lbl).sum().item()
        multi_correct  += (mul_out.argmax(1) == mul_lbl).sum().item()
        total          += len(imgs)
    return total_loss / len(loader), binary_correct / total, multi_correct / total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = binary_correct = multi_correct = total = 0
    all_bin_preds, all_bin_labels = [], []
    all_mul_preds, all_mul_labels = [], []
    with torch.no_grad():
        for imgs, bin_lbl, mul_lbl in loader:
            imgs, bin_lbl, mul_lbl = imgs.to(device), bin_lbl.to(device), mul_lbl.to(device)
            bin_out, mul_out = model(imgs)
            loss = criterion(bin_out, bin_lbl) + criterion(mul_out, mul_lbl)
            total_loss     += loss.item()
            binary_correct += (bin_out.argmax(1) == bin_lbl).sum().item()
            multi_correct  += (mul_out.argmax(1) == mul_lbl).sum().item()
            total          += len(imgs)
            all_bin_preds.extend(bin_out.argmax(1).cpu().numpy())
            all_bin_labels.extend(bin_lbl.cpu().numpy())
            all_mul_preds.extend(mul_out.argmax(1).cpu().numpy())
            all_mul_labels.extend(mul_lbl.cpu().numpy())
    return (total_loss / len(loader), binary_correct / total, multi_correct / total,
            all_bin_preds, all_bin_labels, all_mul_preds, all_mul_labels)

# ─────────────────────────────────────────────
# Run one experiment
# ─────────────────────────────────────────────
def run_experiment(name, all_samples, class_to_idx, idx_to_class, num_classes):
    print(f"\n{'='*60}")
    print(f"  Experiment: {name}")
    print(f"  Samples: {len(all_samples)}")
    print(f"{'='*60}")

    if len(all_samples) == 0:
        print("  [!] No samples found, skipping.")
        return None

    exp_dir = OUTPUT_DIR / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_s, temp_s = train_test_split(all_samples, test_size=0.3, random_state=SEED,
                                        stratify=[s[1] for s in all_samples])
    val_s, test_s   = train_test_split(temp_s, test_size=0.5, random_state=SEED,
                                        stratify=[s[1] for s in temp_s])
    print(f"  Train: {len(train_s)}, Val: {len(val_s)}, Test: {len(test_s)}")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_loader = DataLoader(TrafficImageDataset(train_s, class_to_idx, transform),
                              batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(TrafficImageDataset(val_s,   class_to_idx, transform),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(TrafficImageDataset(test_s,  class_to_idx, transform),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = DualHeadModel(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\n  {'Ep':>4} {'TrLoss':>8} {'TrBin':>8} {'TrMul':>8} {'VaLoss':>8} {'VaBin':>8} {'VaMul':>8}")
    print(f"  {'-'*56}")

    history = defaultdict(list)
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_bin, tr_mul = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        va_loss, va_bin, va_mul, _, _, _, _ = eval_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_binary_acc"].append(tr_bin)
        history["val_binary_acc"].append(va_bin)
        history["train_multi_acc"].append(tr_mul)
        history["val_multi_acc"].append(va_mul)

        print(f"  {epoch:>4} {tr_loss:>8.4f} {tr_bin:>8.4f} {tr_mul:>8.4f} {va_loss:>8.4f} {va_bin:>8.4f} {va_mul:>8.4f}")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(model.state_dict(), exp_dir / "best_model.pt")

    # Evaluate on test set
    model.load_state_dict(torch.load(exp_dir / "best_model.pt"))
    _, _, _, bin_preds, bin_labels, mul_preds, mul_labels = eval_epoch(
        model, test_loader, criterion, DEVICE)

    bin_acc = accuracy_score(bin_labels, bin_preds)
    bin_f1  = f1_score(bin_labels, bin_preds, average="macro")
    mul_acc = accuracy_score(mul_labels, mul_preds)
    mul_f1  = f1_score(mul_labels, mul_preds, average="macro")

    print(f"\n  Test Results:")
    print(f"  Binary  - Accuracy: {bin_acc:.4f}, Macro F1: {bin_f1:.4f}")
    print(f"  Multi   - Accuracy: {mul_acc:.4f}, Macro F1: {mul_f1:.4f}")

    # Confusion matrices
    for labels, preds, cnames, title, fname in [
        (bin_labels, bin_preds, ["benign","malicious"], f"{name} Binary CM", "confusion_binary.png"),
        (mul_labels, mul_preds, [idx_to_class[i] for i in range(num_classes)], f"{name} Multi-class CM", "confusion_multiclass.png"),
    ]:
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(max(8, len(cnames)), max(6, len(cnames)-2)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=cnames, yticklabels=cnames)
        plt.title(title)
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(exp_dir / fname, dpi=150)
        plt.close()

    # Training history plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Training History - {name}", fontsize=14)
    pairs  = [("train_loss","val_loss"), ("train_binary_acc","val_binary_acc"), ("train_multi_acc","val_multi_acc")]
    titles = ["Loss", "Binary Accuracy", "Multi-class Accuracy"]
    for ax, (k1, k2), t in zip(axes, pairs, titles):
        ax.plot(history[k1], label="Train")
        ax.plot(history[k2], label="Val")
        ax.set_title(t)
        ax.legend()
    plt.tight_layout()
    plt.savefig(exp_dir / "training_history.png", dpi=150)
    plt.close()

    print(f"  Plots saved to {exp_dir}")

    return {
        "name":     name,
        "samples":  len(all_samples),
        "bin_acc":  round(bin_acc, 4),
        "bin_f1":   round(bin_f1, 4),
        "mul_acc":  round(mul_acc, 4),
        "mul_f1":   round(mul_f1, 4),
    }

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("\nCollecting samples...")

    packet_grey = collect_images(PACKET_DIR, "greyscale", CAP_PACKET)
    packet_rgb  = collect_images(PACKET_DIR, "rgb",       CAP_PACKET)
    flow_grey   = collect_images(FLOW_DIR,   "greyscale", CAP_FLOW)
    flow_rgb    = collect_images(FLOW_DIR,   "rgb",       CAP_FLOW)

    print(f"  Packet greyscale: {len(packet_grey)} | Packet RGB: {len(packet_rgb)}")
    print(f"  Flow   greyscale: {len(flow_grey)}   | Flow   RGB: {len(flow_rgb)}")

    # Build shared class mapping from all samples combined
    all_samples  = packet_grey + packet_rgb + flow_grey + flow_rgb
    classes      = sorted(set(s[2] for s in all_samples))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    num_classes  = len(classes)
    print(f"Classes ({num_classes}): {classes}")

    experiments = [
        ("Packet_Greyscale", packet_grey),
        ("Packet_RGB",       packet_rgb),
        ("Flow_Greyscale",   flow_grey),
        ("Flow_RGB",         flow_rgb),
        ("Combined_Greyscale", packet_grey + flow_grey),
        ("Combined_RGB",       packet_rgb  + flow_rgb),
    ]

    results = []
    for name, samples in experiments:
        result = run_experiment(name, samples, class_to_idx, idx_to_class, num_classes)
        if result:
            results.append(result)

    # ── Summary ────────────────────────────────
    print("\n" + "="*60)
    print("  FINAL COMPARISON SUMMARY")
    print("="*60)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv(OUTPUT_DIR / "comparison_results.csv", index=False)

    # Comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Greyscale vs RGB - All Experiments", fontsize=14)
    names = [r["name"] for r in results]
    x = range(len(names))
    colors = ["#2196F3","#1565C0","#FF9800","#E65100","#4CAF50","#2E7D32"]

    for ax, metric_acc, metric_f1, title in [
        (axes[0], "bin_acc", "bin_f1",  "Binary Classification"),
        (axes[1], "mul_acc", "mul_f1",  "Multi-class Classification"),
    ]:
        bars = ax.bar(x, [r[metric_acc] for r in results], color=colors, label="Accuracy")
        ax.bar(x, [r[metric_f1] for r in results], color=colors, alpha=0.4, label="Macro F1")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylim(0.5, 1.01)
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_chart.png", dpi=150)
    plt.close()
    print(f"\nComparison chart saved to {OUTPUT_DIR / 'comparison_chart.png'}")
    print("\nDone")

if __name__ == "__main__":
    main()