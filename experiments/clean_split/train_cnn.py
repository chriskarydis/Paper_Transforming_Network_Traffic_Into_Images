#!/usr/bin/env python3
import sys
import json
import random
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score

SEED = 42
IMG_SIZE = 32
BATCH_SIZE = 128
EPOCHS = 15
LR = 1e-3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CNN_CLASSES = [
    "0day", "blackEnergy", "distcc_exec_backdoor",
    "hydra_ftp", "hydra_ssh", "java_rmi",
    "mirai", "netbios_ssn", "normal_browsing",
    "replayAttacks", "ruby_drb", "smtp",
    "tomcat", "unreallrcd", "vsftpd", "zeus",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CNN_CLASSES)}
NUM_CLASSES = len(CNN_CLASSES)


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


class ImgDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, bin_lbl, mul_lbl = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), 0)
        return self.transform(img), bin_lbl, mul_lbl


def collect_split(root: Path, split: str):
    samples = []
    split_dir = root / split
    if not split_dir.exists():
        return samples
    for cls_dir in sorted(split_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name
        if cls not in CLASS_TO_IDX:
            continue
        bin_lbl = 0 if cls == "normal_browsing" else 1
        mul_lbl = CLASS_TO_IDX[cls]
        for img_path in cls_dir.glob("*.png"):
            samples.append((str(img_path), bin_lbl, mul_lbl))
    return samples


def run_epoch(model, loader, optimizer, criterion, train=True):
    model.train() if train else model.eval()
    total_loss = bin_correct = mul_correct = total = 0
    all_mul_preds, all_mul_labels = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, bin_lbl, mul_lbl in loader:
            imgs, bin_lbl, mul_lbl = imgs.to(DEVICE), bin_lbl.to(DEVICE), mul_lbl.to(DEVICE)
            if train:
                optimizer.zero_grad()
            bin_out, mul_out = model(imgs)
            loss = criterion(bin_out, bin_lbl) + criterion(mul_out, mul_lbl)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            bin_correct += (bin_out.argmax(1) == bin_lbl).sum().item()
            mul_correct += (mul_out.argmax(1) == mul_lbl).sum().item()
            total += len(imgs)
            all_mul_preds.extend(mul_out.argmax(1).cpu().numpy())
            all_mul_labels.extend(mul_lbl.cpu().numpy())
    mul_f1 = f1_score(all_mul_labels, all_mul_preds, average="macro")
    return total_loss / len(loader), bin_correct / total, mul_correct / total, mul_f1


def main():
    ap = ArgumentParser()
    ap.add_argument("--images-root", required=True, help="root dir with train/val/test subdirs")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    root = Path(args.images_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_samples = collect_split(root, "train")
    val_samples = collect_split(root, "val")
    test_samples = collect_split(root, "test")
    print(f"[{args.name}] train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    train_loader = DataLoader(ImgDataset(train_samples, transform), batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(ImgDataset(val_samples, transform), batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(ImgDataset(test_samples, transform), batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    model = DualHeadModel(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1 = -1
    best_state = None
    print(f"{'Ep':>4} {'TrLoss':>8} {'TrBin':>8} {'TrMul':>8} {'VaLoss':>8} {'VaBin':>8} {'VaMul':>8} {'VaF1':>8}")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_bin, tr_mul, tr_f1 = run_epoch(model, train_loader, optimizer, criterion, train=True)
        va_loss, va_bin, va_mul, va_f1 = run_epoch(model, val_loader, optimizer, criterion, train=False)
        scheduler.step()
        print(f"{epoch:>4} {tr_loss:>8.4f} {tr_bin:>8.4f} {tr_mul:>8.4f} {va_loss:>8.4f} {va_bin:>8.4f} {va_mul:>8.4f} {va_f1:>8.4f}")
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "best_model.pt")
    print(f"\nBest val macro F1: {best_val_f1:.4f}")

    # Final test evaluation
    te_loss, te_bin, te_mul, te_f1 = run_epoch(model, test_loader, optimizer, criterion, train=False)
    print(f"Test: bin_acc={te_bin:.4f} mul_acc={te_mul:.4f} mul_f1={te_f1:.4f}")

    # Detailed test metrics (bin_f1, AUC etc.) computed via full predict pass
    model.eval()
    all_bin_probs, all_bin_labels = [], []
    all_mul_probs, all_mul_labels = [], []
    with torch.no_grad():
        for imgs, bin_lbl, mul_lbl in test_loader:
            imgs = imgs.to(DEVICE)
            bin_out, mul_out = model(imgs)
            all_bin_probs.append(torch.softmax(bin_out, 1).cpu().numpy())
            all_mul_probs.append(torch.softmax(mul_out, 1).cpu().numpy())
            all_bin_labels.extend(bin_lbl.numpy())
            all_mul_labels.extend(mul_lbl.numpy())
    all_bin_probs = np.concatenate(all_bin_probs)
    all_mul_probs = np.concatenate(all_mul_probs)
    np.savez(out_dir / "test_predictions.npz",
             bin_probs=all_bin_probs, bin_labels=np.array(all_bin_labels),
             mul_probs=all_mul_probs, mul_labels=np.array(all_mul_labels))

    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "name": args.name, "n_train": len(train_samples), "n_val": len(val_samples),
            "n_test": len(test_samples), "best_val_mul_f1": best_val_f1,
            "test_bin_acc": te_bin, "test_mul_acc": te_mul, "test_mul_f1": te_f1,
        }, f, indent=2)
    print(f"Saved model + predictions to {out_dir}")


if __name__ == "__main__":
    main()
