#!/usr/bin/env python3
"""
evaluate_ensemble.py

Evaluates the full 5-model weighted soft-voting ensemble on a held-out test set.

Data pipeline (for context):
  data/raw/<class>.pcap
    ├─ heiFIP PacketImage ──► data/images_packet/  (source for Packet CNN training)
    └─ split_flows.py ──► data/flows/<class>/*.pcap
                              ├─ heiFIP FlowImage ──► data/images_flow/ (Flow CNN training)
                              └─ dpkt features ──────► LightGBM training

Evaluation unit: individual flow pcap files from data/flows/
  - Flow images (greyscale + RGB): matched via pcap.stem + "_0.png" (pre-existing)
  - Packet images: generated on-the-fly from each flow pcap via heiFIP PacketImage
  - LightGBM features: extracted via dpkt

CAVEATS:
  - Packet CNNs were trained on PacketImages of raw multi-flow captures; evaluating
    them on single-flow PacketImages introduces a mild distribution shift. Treat
    Packet CNN metrics as approximate.
  - LightGBM was trained on all flow pcap data (5-fold CV, no held-out test set).
    Its metrics here are evaluated on training data and will be optimistically biased.
  - The ensemble metric is the primary result of interest.

Models & weights (multi-class macro F1 from individual training):
  1. Packet_Greyscale CNN  w = 0.9868
  2. Packet_RGB CNN        w = 0.9834
  3. Flow_Greyscale CNN    w = 0.8303
  4. Flow_RGB CNN          w = 0.9108
  5. LightGBM              w = 0.8635

Outputs (saved to models/ensemble_eval/):
  - metrics_summary.csv       - per-model + ensemble accuracy & macro F1
  - raw_results.csv           - per-sample predictions for every model
  - confusion_binary.png      - ensemble binary confusion matrix
  - confusion_multiclass.png  - ensemble multi-class confusion matrix
  - classification_report.txt - per-class precision / recall / F1 for ensemble
"""

import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "1")

_TABLES_ONLY = "--tables-only" in sys.argv

import json
import random
import subprocess
import tempfile
import time
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib
import dpkt
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, roc_auc_score,
    classification_report, confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE         = Path.home() / "thesis_heifip"
FLOWS_DIR    = BASE / "data" / "flows"
IMAGES_FLOW  = BASE / "data" / "images_flow"
HEIFIP       = BASE / "tools" / "heiFIP" / "heiFIP" / "build" / "heiFIP"
OUT_DIR      = BASE / "models" / "ensemble_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATHS = {
    "pkt_grey":  BASE / "models" / "comparison" / "Packet_Greyscale"    / "best_model.pt",
    "pkt_rgb":   BASE / "models" / "comparison" / "Packet_RGB"           / "best_model.pt",
    "flw_grey":  BASE / "models" / "comparison" / "Flow_Greyscale"       / "best_model.pt",
    "flw_rgb":   BASE / "models" / "comparison" / "Flow_RGB"             / "best_model.pt",
    "comb_grey": BASE / "models" / "comparison" / "Combined_Greyscale"   / "best_model.pt",
    "comb_rgb":  BASE / "models" / "comparison" / "Combined_RGB"         / "best_model.pt",
}

LGB_BIN_PATH    = BASE / "models" / "branch3" / "binary_model.pkl"
LGB_MUL_PATH    = BASE / "models" / "branch3" / "multi_model.pkl"
LGB_LE_PATH     = BASE / "models" / "branch3" / "label_encoder.pkl"
LGB_FEAT_PATH   = BASE / "models" / "branch3" / "feature_cols.json"
LGB_THRESH_PATH = BASE / "models" / "branch3" / "class_thresholds.npy"

# ─── Config ───────────────────────────────────────────────────────────────────
SEED          = 42
CAP_PER_CLASS = 500       # max samples per class (matches Flow CNN training cap)
TEST_FRAC     = 0.15      # held-out test fraction
IMG_SIZE      = 32
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HEIFIP_TIMEOUT = 30       # seconds per heiFIP call

# Min-Confidence Vote: models below this multiclass confidence are excluded per sample
MIN_CONF_THRESHOLD = 0.9

# Ensemble weights = per-model multi-class macro F1 from training
WEIGHTS = {
    "pkt_grey":  0.9868,
    "pkt_rgb":   0.9834,
    "flw_grey":  0.8303,
    "flw_rgb":   0.9108,
    "comb_grey": 0.9717,
    "comb_rgb":  0.9697,
    "lgb":       0.8635,
}

CNN_CLASSES = [
    "0day", "blackEnergy", "distcc_exec_backdoor",
    "hydra_ftp", "hydra_ssh", "java_rmi",
    "mirai", "netbios_ssn", "normal_browsing",
    "replayAttacks", "ruby_drb", "smtp",
    "tomcat", "unreallrcd", "vsftpd", "zeus",
]
NUM_CLASSES  = len(CNN_CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CNN_CLASSES)}

random.seed(SEED)
np.random.seed(SEED)

HEIFIP_AVAILABLE = HEIFIP.exists()
if not HEIFIP_AVAILABLE:
    print(f"[!] heiFIP binary not found at {HEIFIP}")
    print("    Packet CNN predictions will use uniform distributions.")

# ─── CNN Model ────────────────────────────────────────────────────────────────
class DualHeadModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        backbone         = models.resnet18(weights=None)
        backbone.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        in_features      = backbone.fc.in_features
        backbone.fc      = nn.Identity()
        self.backbone    = backbone
        self.binary_head = nn.Linear(in_features, 2)
        self.multi_head  = nn.Linear(in_features, num_classes)

    def forward(self, x):
        f = self.backbone(x)
        return self.binary_head(f), self.multi_head(f)


TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


def load_cnn(path: Path) -> DualHeadModel:
    m = DualHeadModel(NUM_CLASSES).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m


if not _TABLES_ONLY:
    print("Loading CNN models...")
    CNN_MODELS = {k: load_cnn(p) for k, p in MODEL_PATHS.items()}

    print("Loading LightGBM models...")
    LGB_BIN     = joblib.load(LGB_BIN_PATH)
    LGB_MUL     = joblib.load(LGB_MUL_PATH)
    LGB_LE      = joblib.load(LGB_LE_PATH)
    LGB_FEATS   = json.load(open(LGB_FEAT_PATH))
    LGB_THRESH  = np.load(LGB_THRESH_PATH) if LGB_THRESH_PATH.exists() else None
    LGB_CLASSES = list(LGB_LE.classes_)

    # Precompute LGB → CNN class index alignment
    REORDER_IDX = [
        LGB_CLASSES.index(c) if c in LGB_CLASSES else -1
        for c in CNN_CLASSES
    ]

# ─── Feature Extraction ───────────────────────────────────────────────────────
def extract_features(pcap_path: str) -> dict | None:
    """Per-flow statistical feature extraction - mirrors analyze.py exactly."""
    try:
        f    = open(pcap_path, "rb")
        pcap = dpkt.pcap.Reader(f)
    except Exception:
        return None

    sizes, timestamps, iats          = [], [], []
    header_lens, ttls, proto_types   = [], [], []
    dst_ports, src_ports             = [], []
    fin_count = syn_count = rst_count = 0
    psh_count = ack_count = ece_count = cwr_count = 0
    http = https = dns = telnet = smtp_f = ssh = irc = 0
    tcp  = udp   = dhcp = arp   = icmp  = igmp = ipv = 0
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
                icmp = 1; continue
            if isinstance(ip.data, dpkt.igmp.IGMP):
                igmp = 1; continue
            if isinstance(ip.data, dpkt.udp.UDP):
                u = ip.data; udp = 1
                header_lens.append(8)
                sport, dport = u.sport, u.dport
                dst_ports.append(dport); src_ports.append(sport)
                if dport == 53 or sport == 53:                dns    = 1
                if dport == 67 or dport == 68:                dhcp   = 1
                if dport in (137, 138) or sport in (137, 138): netbios = 1
            elif isinstance(ip.data, dpkt.tcp.TCP):
                t = ip.data; tcp = 1
                header_lens.append(t.off * 4)
                sport, dport = t.sport, t.dport
                dst_ports.append(dport); src_ports.append(sport)
                flags = t.flags
                if flags & dpkt.tcp.TH_FIN:  fin_count += 1
                if flags & dpkt.tcp.TH_SYN:  syn_count += 1
                if flags & dpkt.tcp.TH_RST:  rst_count += 1
                if flags & dpkt.tcp.TH_PUSH: psh_count += 1
                if flags & dpkt.tcp.TH_ACK:  ack_count += 1
                if flags & 0x40:             ece_count += 1
                if flags & 0x80:             cwr_count += 1
                if dport == 80   or sport == 80:   http   = 1
                if dport == 443  or sport == 443:  https  = 1
                if dport == 22   or sport == 22:   ssh    = 1
                if dport == 23   or sport == 23:   telnet = 1
                if dport == 25   or sport == 25:   smtp_f = 1
                if dport in (194, 6667, 6668, 6697) or \
                   sport in (194, 6667, 6668, 6697): irc   = 1
                if dport in (20, 21) or sport in (20, 21):      ftp     = 1
                if dport in (139, 445) or sport in (139, 445):  netbios = 1
    except Exception:
        pass
    finally:
        f.close()

    if not sizes:
        return None

    s   = np.array(sizes, dtype=float)
    ia  = np.array(iats,  dtype=float) if iats else np.array([0.0])
    dur = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
    sq25, sq50, sq75 = np.percentile(s,  [25, 50, 75])
    iq25, iq50, iq75 = np.percentile(ia, [25, 50, 75])
    _mode = lambda lst: max(set(lst), key=lst.count) if lst else 0

    return {
        "pkt_count":          count,
        "duration":           dur,
        "rate":               count / (dur + 1e-9),
        "bytes_per_sec":      s.sum() / (dur + 1e-9),
        "tot_size":           s.sum(),
        "min_size":           s.min(),
        "max_size":           s.max(),
        "avg_size":           s.mean(),
        "std_size":           s.std(),
        "var_size":           s.var(),
        "size_range":         s.max() - s.min(),
        "size_q25":           float(sq25),
        "size_q50":           float(sq50),
        "size_q75":           float(sq75),
        "iat_mean":           ia.mean(),
        "iat_std":            ia.std(),
        "iat_min":            ia.min(),
        "iat_max":            ia.max(),
        "iat_var":            ia.var(),
        "iat_q25":            float(iq25),
        "iat_q50":            float(iq50),
        "iat_q75":            float(iq75),
        "header_len_mean":    float(np.mean(header_lens)) if header_lens else 0.0,
        "ttl_mean":           float(np.mean(ttls))        if ttls else 0.0,
        "ttl_std":            float(np.std(ttls))         if ttls else 0.0,
        "proto_type_mode":    int(_mode(proto_types)),
        "dst_port_mode":      int(_mode(dst_ports)),
        "src_port_mode":      int(_mode(src_ports)),
        "n_unique_dst_ports": len(set(dst_ports)),
        "n_unique_src_ports": len(set(src_ports)),
        "bytes_per_pkt":      s.sum() / (count + 1e-9),
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
        "f_smtp":             smtp_f,
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

# ─── CNN Inference ────────────────────────────────────────────────────────────
_UNIFORM_BIN = np.array([0.5, 0.5])
_UNIFORM_MUL = np.ones(NUM_CLASSES) / NUM_CLASSES


def cnn_proba(image_paths: list, model: DualHeadModel):
    """Returns (avg_bin_proba[2], avg_mul_proba[NUM_CLASSES]) over all images."""
    tensors = []
    for p in image_paths:
        try:
            tensors.append(TRANSFORM(Image.open(p).convert("RGB")))
        except Exception:
            pass
    if not tensors:
        return _UNIFORM_BIN.copy(), _UNIFORM_MUL.copy()
    batch = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        b_out, m_out = model(batch)
    b = torch.softmax(b_out, dim=1).cpu().numpy()
    m = torch.softmax(m_out, dim=1).cpu().numpy()
    return b.mean(axis=0), m.mean(axis=0)

# ─── LightGBM Inference ───────────────────────────────────────────────────────
def lgb_proba(pcap_path: Path):
    """
    Returns (bin_proba[2], mul_proba[NUM_CLASSES]) in CNN_CLASSES order,
    or (None, None) if feature extraction fails.
    Missing features (has_port_*) are filled with 0, matching analyze.py behaviour.
    """
    feats = extract_features(str(pcap_path))
    if feats is None:
        return None, None

    df = pd.DataFrame([feats])
    df = df.reindex(columns=LGB_FEATS, fill_value=0)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    X = df.values

    bin_p = LGB_BIN.predict_proba(X)[0]   # [P(benign), P(malicious)]
    mul_p = LGB_MUL.predict_proba(X)[0]   # in LGB_LE class order

    if LGB_THRESH is not None:
        mul_p = mul_p * LGB_THRESH
        s = mul_p.sum()
        if s > 0:
            mul_p /= s

    # Reorder to match CNN_CLASSES
    aligned = np.zeros(NUM_CLASSES)
    for cnn_i, lgb_i in enumerate(REORDER_IDX):
        if lgb_i >= 0:
            aligned[cnn_i] = mul_p[lgb_i]
    s = aligned.sum()
    if s > 0:
        aligned /= s

    return bin_p, aligned

# ─── heiFIP Packet Image Generation ──────────────────────────────────────────
def generate_packet_images(pcap_path: Path, out_dir: Path, rgb: bool) -> list:
    """
    Run heiFIP in PacketImage mode on a flow pcap and return all generated PNGs.
    Note: Packet CNN models were trained on raw multi-flow captures; using single-flow
    pcaps here is a mild distribution shift but allows per-sample ensemble evaluation.
    """
    if not HEIFIP_AVAILABLE:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pcap_path.stem
    cmd  = [
        str(HEIFIP), "-i", str(pcap_path), "-o", str(out_dir),
        "-m", "PacketImage", "--min-pkts", "1", "--name", stem,
    ]
    if rgb:
        cmd.append("--rgb")
    try:
        subprocess.run(cmd, capture_output=True, timeout=HEIFIP_TIMEOUT)
    except (subprocess.TimeoutExpired, Exception):
        pass
    return list(out_dir.glob(f"{stem}*.png"))

# ─── Dataset Collection ───────────────────────────────────────────────────────
def collect_samples() -> list[dict]:
    """
    Walk data/flows/, cap at CAP_PER_CLASS per class, keep only samples where
    both greyscale and RGB flow images already exist (pcap.stem + "_0.png").
    """
    samples      : list[dict]       = []
    class_counts : dict[str, int]   = defaultdict(int)

    for cat_dir in sorted(FLOWS_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        is_mal  = "malicious" in cat_dir.name
        bin_lbl = 1 if is_mal else 0

        for cls_dir in sorted(cat_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls   = cls_dir.name
            pcaps = list(cls_dir.glob("*.pcap"))
            random.shuffle(pcaps)

            for pcap in pcaps:
                if class_counts[cls] >= CAP_PER_CLASS:
                    break
                stem     = pcap.stem
                grey_img = IMAGES_FLOW / "greyscale" / cat_dir.name / cls / f"{stem}_0.png"
                rgb_img  = IMAGES_FLOW / "rgb"       / cat_dir.name / cls / f"{stem}_0.png"
                if grey_img.exists() and rgb_img.exists():
                    samples.append({
                        "pcap":     pcap,
                        "grey_img": grey_img,
                        "rgb_img":  rgb_img,
                        "class":    cls,
                        "binary":   bin_lbl,
                    })
                    class_counts[cls] += 1

    active = {cls: n for cls, n in class_counts.items() if n > 0}
    print(f"\nCollected {len(samples)} matched samples across {len(active)} classes:")
    for cls, n in sorted(active.items()):
        print(f"  {cls:<32} {n}")
    return samples

# ─── Metrics Table PNG ────────────────────────────────────────────────────────
def _save_metrics_table_png(metrics_rows: list[dict], out_path: Path, task: str):
    """Render a publication-style metrics table to a PNG file."""
    import matplotlib.patches as mpatches

    GROUPS = [
        ("Packet\nBranch",    ["Packet_Greyscale",   "Packet_RGB"]),
        ("Flow\nBranch",      ["Flow_Greyscale",     "Flow_RGB"]),
        ("Combined\nBranch",  ["Combined_Greyscale", "Combined_RGB"]),
        ("LightGBM\nBranch",  ["LightGBM *"]),
        ("Ensemble",          ["Weighted Soft-Vote", "Max-Confidence Vote", "Min-Confidence Vote"]),
        ("Stacking",          ["Meta-LightGBM"]),
    ]
    DISPLAY = {
        "Packet_Greyscale":   "Greyscale",
        "Packet_RGB":         "RGB",
        "Flow_Greyscale":     "Greyscale",
        "Flow_RGB":           "RGB",
        "Combined_Greyscale": "Greyscale",
        "Combined_RGB":       "RGB",
        "LightGBM *":         "LightGBM",
        "Weighted Soft-Vote":  "WSV",
        "Max-Confidence Vote": "MCV",
        "Min-Confidence Vote": "Min-CV",
        "Meta-LightGBM":       "Meta-LightGBM",
    }
    _GC = ["#f0f4f8", "#eaf4fb", "#fdf2f8", "#fef9e7", "#e9f7ef", "#f5eef8"]
    GROUP_COLORS = _GC  # indexed by g_idx_actual; cycles if more groups than colours

    if task == "binary":
        mk = ["bin_acc", "bin_f1", "bin_prec", "bin_rec", "bin_mcc", "bin_auc"]
        title = "Binary Classification - Malicious vs. Benign"
    else:
        mk = ["mul_acc", "mul_f1", "mul_prec", "mul_rec", "mul_mcc", "mul_auc"]
        title = "Multiclass Classification - Attack Type"

    col_hdrs = ["Approaches", "Model", "Accuracy", "F1-score",
                "Precision", "Recall", "MCC", "ROC-AUC"]
    col_w    = [1.9, 1.3, 1.5, 1.5, 1.5, 1.5, 1.3, 1.5]  # relative widths
    total_w  = sum(col_w)
    n_cols   = len(col_hdrs)

    df_m = pd.DataFrame(metrics_rows).set_index("model")
    best = {k: df_m[k].max() for k in mk}

    # Flatten rows in group order, skipping groups not present in metrics_rows
    rows = []
    g_idx_actual = 0
    for gname, members in GROUPS:
        present = [m for m in members if m in df_m.index]
        if not present:
            continue
        for i, model in enumerate(present):
            rows.append({
                "group":   gname if i == 0 else "",
                "display": DISPLAY[model],
                "model":   model,
                "g_idx":   g_idx_actual,
                "vals":    [df_m.loc[model, k] for k in mk],
            })
        g_idx_actual += 1

    n_rows   = len(rows)
    cell_h   = 0.55          # inches per data row
    header_h = 0.65          # inches for header row
    fig_w    = 13.0
    fig_h    = header_h + n_rows * cell_h + 0.5

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    # Cumulative x positions
    xs = [0.0]
    for w in col_w:
        xs.append(xs[-1] + w)

    def cell(x0, x1, y0, h, fc, ec="#cccccc", lw=0.5):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y0), x1 - x0, h,
            boxstyle="square,pad=0", linewidth=lw,
            facecolor=fc, edgecolor=ec, transform=ax.transData,
        ))

    def txt(x, y, s, **kw):
        kw.setdefault("ha", "center")
        kw.setdefault("va", "center")
        kw.setdefault("fontsize", 9)
        ax.text(x, y, s, **kw)

    # Header row (top of figure)
    hy0 = fig_h - header_h
    for j in range(n_cols):
        cell(xs[j], xs[j+1], hy0, header_h, fc="#2c3e50", ec="#2c3e50")
        txt((xs[j] + xs[j+1]) / 2, hy0 + header_h / 2, col_hdrs[j],
            color="white", fontweight="bold")

    # Draw data rows bottom-up
    group_tops = {}  # track y-top of first row in each group (for separator lines)
    for i, row in enumerate(rows):
        y0  = fig_h - header_h - (i + 1) * cell_h
        mid = y0 + cell_h / 2
        gc  = GROUP_COLORS[row["g_idx"] % len(GROUP_COLORS)]

        # Background for all cells in this row
        for j in range(n_cols):
            cell(xs[j], xs[j+1], y0, cell_h, fc=gc)

        # Group label (col 0)
        txt((xs[0] + xs[1]) / 2, mid, row["group"], fontweight="bold", fontsize=8.5)
        # Model label (col 1)
        txt((xs[1] + xs[2]) / 2, mid, row["display"], fontsize=8.5)

        # Metric values
        for k_idx, (val, key) in enumerate(zip(row["vals"], mk)):
            j   = k_idx + 2
            xc  = (xs[j] + xs[j+1]) / 2
            is_best = abs(val - best[key]) < 1e-9
            if is_best:
                cell(xs[j], xs[j+1], y0, cell_h, fc="#d6eaf8")
            txt(xc, mid, f"{val*100:.2f}%",
                fontweight="bold" if is_best else "normal", fontsize=8.5)

        if row["group"]:  # first row of a group - remember top edge for separator
            group_tops[row["g_idx"]] = y0 + cell_h

    # Horizontal separators between groups
    ax.axhline(hy0, color="#333333", linewidth=1.2)          # below header
    ax.axhline(0,   color="#333333", linewidth=1.2)           # bottom border
    for g_idx in range(1, len(GROUPS)):
        y_sep = group_tops.get(g_idx, None)
        if y_sep is not None:
            ax.axhline(y_sep, color="#555555", linewidth=0.8)

    ax.set_title(title, y=1.0, pad=4, fontsize=11, fontweight="bold")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out_path}")


# ─── Main Evaluation ──────────────────────────────────────────────────────────
def run_evaluation():
    samples = collect_samples()

    # Stratified 15% test split (mirrors the 70/15/15 split used in CNN training)
    _, test_samples = train_test_split(
        samples,
        test_size=TEST_FRAC,
        random_state=SEED,
        stratify=[s["class"] for s in samples],
    )
    print(f"\nTest set: {len(test_samples)} samples  (device: {DEVICE})")

    results : list[dict] = []
    skipped = 0
    t0      = time.time()

    # Full softmax vectors accumulated per sample - written to meta_features.npz
    # for use by train_meta.py (stacking meta-learner training).
    proba_store: dict[str, list] = {k: [] for k in [
        "pkt_grey_mul",  "pkt_grey_bin",
        "pkt_rgb_mul",   "pkt_rgb_bin",
        "flw_grey_mul",  "flw_grey_bin",
        "flw_rgb_mul",   "flw_rgb_bin",
        "comb_grey_mul", "comb_grey_bin",
        "comb_rgb_mul",  "comb_rgb_bin",
        "lgb_mul",       "lgb_bin",
    ]}

    with tempfile.TemporaryDirectory() as _tmp:
        tmpdir = Path(_tmp)

        for i, s in enumerate(test_samples):
            if (i + 1) % 25 == 0:
                elapsed = time.time() - t0
                eta     = elapsed / (i + 1) * (len(test_samples) - i - 1)
                print(f"  [{i+1:>4}/{len(test_samples)}]  "
                      f"elapsed {elapsed/60:.1f} min  ETA {eta/60:.1f} min")

            pcap     = s["pcap"]
            true_cls = s["class"]
            true_bin = s["binary"]
            true_mul = CLASS_TO_IDX.get(true_cls, -1)
            if true_mul < 0:
                skipped += 1
                continue

            # ── Flow CNNs: use pre-existing flow images ───────────────────────
            flw_grey_bin, flw_grey_mul = cnn_proba([s["grey_img"]], CNN_MODELS["flw_grey"])
            flw_rgb_bin,  flw_rgb_mul  = cnn_proba([s["rgb_img"]],  CNN_MODELS["flw_rgb"])

            # ── Packet CNNs: generate images from flow pcap on-the-fly ────────
            sd             = tmpdir / str(i)
            pkt_grey_imgs  = generate_packet_images(pcap, sd / "pg", rgb=False)
            pkt_rgb_imgs   = generate_packet_images(pcap, sd / "pr", rgb=True)
            pkt_grey_bin, pkt_grey_mul = cnn_proba(pkt_grey_imgs, CNN_MODELS["pkt_grey"])
            pkt_rgb_bin,  pkt_rgb_mul  = cnn_proba(pkt_rgb_imgs,  CNN_MODELS["pkt_rgb"])

            # ── Combined CNNs: flow image + packet images together ────────────
            comb_grey_bin, comb_grey_mul = cnn_proba(
                [s["grey_img"]] + pkt_grey_imgs, CNN_MODELS["comb_grey"])
            comb_rgb_bin,  comb_rgb_mul  = cnn_proba(
                [s["rgb_img"]]  + pkt_rgb_imgs,  CNN_MODELS["comb_rgb"])

            # ── LightGBM ──────────────────────────────────────────────────────
            lgb_bin_p, lgb_mul_p = lgb_proba(pcap)
            lgb_ok = lgb_bin_p is not None

            # ── Accumulate full softmax vectors for meta-learner training ─────
            proba_store["pkt_grey_mul"].append(pkt_grey_mul)
            proba_store["pkt_grey_bin"].append(pkt_grey_bin)
            proba_store["pkt_rgb_mul"].append(pkt_rgb_mul)
            proba_store["pkt_rgb_bin"].append(pkt_rgb_bin)
            proba_store["flw_grey_mul"].append(flw_grey_mul)
            proba_store["flw_grey_bin"].append(flw_grey_bin)
            proba_store["flw_rgb_mul"].append(flw_rgb_mul)
            proba_store["flw_rgb_bin"].append(flw_rgb_bin)
            proba_store["comb_grey_mul"].append(comb_grey_mul)
            proba_store["comb_grey_bin"].append(comb_grey_bin)
            proba_store["comb_rgb_mul"].append(comb_rgb_mul)
            proba_store["comb_rgb_bin"].append(comb_rgb_bin)
            proba_store["lgb_mul"].append(
                lgb_mul_p if lgb_ok else np.ones(NUM_CLASSES) / NUM_CLASSES)
            proba_store["lgb_bin"].append(
                lgb_bin_p if lgb_ok else np.array([0.5, 0.5]))

            # ── Weighted soft-vote ensemble ────────────────────────────────────
            w     = WEIGHTS
            lgb_w = w["lgb"] if lgb_ok else 0.0
            w_tot = (w["pkt_grey"] + w["pkt_rgb"] + w["flw_grey"] + w["flw_rgb"]
                     + w["comb_grey"] + w["comb_rgb"] + lgb_w)

            ens_mul = (
                w["pkt_grey"]  * pkt_grey_mul  +
                w["pkt_rgb"]   * pkt_rgb_mul   +
                w["flw_grey"]  * flw_grey_mul  +
                w["flw_rgb"]   * flw_rgb_mul   +
                w["comb_grey"] * comb_grey_mul +
                w["comb_rgb"]  * comb_rgb_mul
            )
            ens_bin = (
                w["pkt_grey"]  * pkt_grey_bin  +
                w["pkt_rgb"]   * pkt_rgb_bin   +
                w["flw_grey"]  * flw_grey_bin  +
                w["flw_rgb"]   * flw_rgb_bin   +
                w["comb_grey"] * comb_grey_bin +
                w["comb_rgb"]  * comb_rgb_bin
            )
            if lgb_ok:
                ens_mul += w["lgb"] * lgb_mul_p
                ens_bin += w["lgb"] * lgb_bin_p
            ens_mul /= w_tot
            ens_bin /= w_tot

            # ── Min-Confidence Vote ───────────────────────────────────────────
            # Only models whose multiclass confidence >= threshold contribute.
            # Falls back to WSV if no model meets the threshold.
            _all = [
                (pkt_grey_bin,  pkt_grey_mul,  pkt_grey_mul.max(),  w["pkt_grey"]),
                (pkt_rgb_bin,   pkt_rgb_mul,   pkt_rgb_mul.max(),   w["pkt_rgb"]),
                (flw_grey_bin,  flw_grey_mul,  flw_grey_mul.max(),  w["flw_grey"]),
                (flw_rgb_bin,   flw_rgb_mul,   flw_rgb_mul.max(),   w["flw_rgb"]),
                (comb_grey_bin, comb_grey_mul, comb_grey_mul.max(), w["comb_grey"]),
                (comb_rgb_bin,  comb_rgb_mul,  comb_rgb_mul.max(),  w["comb_rgb"]),
            ]
            if lgb_ok:
                _all.append((lgb_bin_p, lgb_mul_p, lgb_mul_p.max(), w["lgb"]))
            _qual = [(bp, mp, wt) for bp, mp, conf, wt in _all if conf >= MIN_CONF_THRESHOLD]
            if not _qual:
                _qual = [(bp, mp, wt) for bp, mp, _, wt in _all]
            _wt   = sum(wt for _, _, wt in _qual)
            mincv_mul = sum(wt * mp for _, mp, wt in _qual) / _wt
            mincv_bin = sum(wt * bp for bp, _, wt in _qual) / _wt

            results.append({
                "true_class":         true_cls,
                "true_binary":        true_bin,
                "true_mul":           true_mul,
                # Per-model multi-class argmax predictions
                "pkt_grey_mul_pred":  int(pkt_grey_mul.argmax()),
                "pkt_rgb_mul_pred":   int(pkt_rgb_mul.argmax()),
                "flw_grey_mul_pred":  int(flw_grey_mul.argmax()),
                "flw_rgb_mul_pred":   int(flw_rgb_mul.argmax()),
                "comb_grey_mul_pred": int(comb_grey_mul.argmax()),
                "comb_rgb_mul_pred":  int(comb_rgb_mul.argmax()),
                "lgb_mul_pred":       int(lgb_mul_p.argmax()) if lgb_ok else -1,
                # Per-model binary argmax predictions
                "pkt_grey_bin_pred":  int(pkt_grey_bin.argmax()),
                "pkt_rgb_bin_pred":   int(pkt_rgb_bin.argmax()),
                "flw_grey_bin_pred":  int(flw_grey_bin.argmax()),
                "flw_rgb_bin_pred":   int(flw_rgb_bin.argmax()),
                "comb_grey_bin_pred": int(comb_grey_bin.argmax()),
                "comb_rgb_bin_pred":  int(comb_rgb_bin.argmax()),
                "lgb_bin_pred":       int(lgb_bin_p.argmax()) if lgb_ok else -1,
                # Per-model multi-class confidence (max probability) - used for MCV
                "pkt_grey_mul_conf":  float(pkt_grey_mul.max()),
                "pkt_rgb_mul_conf":   float(pkt_rgb_mul.max()),
                "flw_grey_mul_conf":  float(flw_grey_mul.max()),
                "flw_rgb_mul_conf":   float(flw_rgb_mul.max()),
                "comb_grey_mul_conf": float(comb_grey_mul.max()),
                "comb_rgb_mul_conf":  float(comb_rgb_mul.max()),
                "lgb_mul_conf":       float(lgb_mul_p.max()) if lgb_ok else -1.0,
                # P(malicious) from binary head - used for proper ROC-AUC
                "pkt_grey_bin_score":  float(pkt_grey_bin[1]),
                "pkt_rgb_bin_score":   float(pkt_rgb_bin[1]),
                "flw_grey_bin_score":  float(flw_grey_bin[1]),
                "flw_rgb_bin_score":   float(flw_rgb_bin[1]),
                "comb_grey_bin_score": float(comb_grey_bin[1]),
                "comb_rgb_bin_score":  float(comb_rgb_bin[1]),
                "lgb_bin_score":       float(lgb_bin_p[1]) if lgb_ok else -1.0,
                # Weighted soft-vote ensemble predictions + binary score
                "ens_mul_pred":        int(ens_mul.argmax()),
                "ens_bin_pred":        int(ens_bin.argmax()),
                "ens_bin_score":       float(ens_bin[1]),
                # Min-Confidence Vote predictions + binary score
                "mincv_mul_pred":      int(mincv_mul.argmax()),
                "mincv_bin_pred":      int(mincv_bin.argmax()),
                "mincv_bin_score":     float(mincv_bin[1]),
                "lgb_available":       lgb_ok,
            })

    if skipped:
        print(f"[!] Skipped {skipped} samples (unrecognised class).")

    df = pd.DataFrame(results)

    # Save full probability vectors for stacking meta-learner (train_meta.py)
    np.savez(
        OUT_DIR / "meta_features.npz",
        true_binary = df["true_binary"].values,
        true_mul    = df["true_mul"].values,
        **{k: np.array(v) for k, v in proba_store.items()},
    )
    print(f"Saved meta_features.npz  ({len(df)} samples × 6 CNNs + LightGBM)")

    # ── Max-confidence voting: the model with the highest max(mul_proba) decides ──
    # LGB unavailable rows already have lgb_mul_conf = -1.0, so they never win.
    conf_cols = [
        "pkt_grey_mul_conf", "pkt_rgb_mul_conf",
        "flw_grey_mul_conf", "flw_rgb_mul_conf",
        "comb_grey_mul_conf", "comb_rgb_mul_conf",
        "lgb_mul_conf",
    ]
    mul_pred_cols = [
        "pkt_grey_mul_pred", "pkt_rgb_mul_pred",
        "flw_grey_mul_pred", "flw_rgb_mul_pred",
        "comb_grey_mul_pred", "comb_rgb_mul_pred",
        "lgb_mul_pred",
    ]
    bin_pred_cols = [
        "pkt_grey_bin_pred", "pkt_rgb_bin_pred",
        "flw_grey_bin_pred", "flw_rgb_bin_pred",
        "comb_grey_bin_pred", "comb_rgb_bin_pred",
        "lgb_bin_pred",
    ]
    bin_score_cols = [
        "pkt_grey_bin_score", "pkt_rgb_bin_score",
        "flw_grey_bin_score", "flw_rgb_bin_score",
        "comb_grey_bin_score", "comb_rgb_bin_score",
        "lgb_bin_score",
    ]

    conf_matrix      = df[conf_cols].values
    mul_pred_matrix  = df[mul_pred_cols].values
    bin_pred_matrix  = df[bin_pred_cols].values
    bin_score_matrix = df[bin_score_cols].values
    winner_idx       = conf_matrix.argmax(axis=1)

    df["mcv_mul_pred"]    = mul_pred_matrix[np.arange(len(df)), winner_idx]
    df["mcv_bin_pred"]    = bin_pred_matrix[np.arange(len(df)), winner_idx]
    df["mcv_winner_conf"] = conf_matrix[np.arange(len(df)), winner_idx]
    df["mcv_winner_model"] = np.array(
        ["pkt_grey", "pkt_rgb", "flw_grey", "flw_rgb", "comb_grey", "comb_rgb", "lgb"]
    )[winner_idx]
    df["mcv_bin_score"] = bin_score_matrix[np.arange(len(df)), winner_idx]

    df.to_csv(OUT_DIR / "raw_results.csv", index=False)

    true_mul_arr = df["true_mul"].values
    true_bin_arr = df["true_binary"].values

    # ── Per-model + ensemble metrics table ────────────────────────────────────
    model_cols = [
        ("Packet_Greyscale",    "pkt_grey_mul_pred",  "pkt_grey_bin_pred",  "pkt_grey_bin_score"),
        ("Packet_RGB",          "pkt_rgb_mul_pred",   "pkt_rgb_bin_pred",   "pkt_rgb_bin_score"),
        ("Flow_Greyscale",      "flw_grey_mul_pred",  "flw_grey_bin_pred",  "flw_grey_bin_score"),
        ("Flow_RGB",            "flw_rgb_mul_pred",   "flw_rgb_bin_pred",   "flw_rgb_bin_score"),
        ("Combined_Greyscale",  "comb_grey_mul_pred", "comb_grey_bin_pred", "comb_grey_bin_score"),
        ("Combined_RGB",        "comb_rgb_mul_pred",  "comb_rgb_bin_pred",  "comb_rgb_bin_score"),
        ("LightGBM *",          "lgb_mul_pred",       "lgb_bin_pred",       "lgb_bin_score"),
        ("Weighted Soft-Vote",    "ens_mul_pred",   "ens_bin_pred",   "ens_bin_score"),
        ("Max-Confidence Vote",   "mcv_mul_pred",   "mcv_bin_pred",   "mcv_bin_score"),
        ("Min-Confidence Vote",   "mincv_mul_pred", "mincv_bin_pred", "mincv_bin_score"),
    ]

    KW = dict(average="macro", zero_division=0)

    print("\n" + "=" * 100)
    print(f"{'Model':<24} {'N':>5}  "
          f"{'BinAcc':>7} {'BinF1':>7} {'BinPrec':>8} {'BinRec':>7} {'BinMCC':>7} {'BinAUC':>7}  "
          f"{'MulAcc':>7} {'MulF1':>7} {'MulPrec':>8} {'MulRec':>7} {'MulMCC':>7} {'MulAUC':>7}")
    print("=" * 100)

    metrics_rows = []
    for name, mul_col, bin_col, score_col in model_cols:
        valid      = df[mul_col] >= 0
        mul_preds  = df.loc[valid, mul_col].values
        bin_preds  = df.loc[valid, bin_col].values
        bin_scores = df.loc[valid, score_col].values
        true_m     = true_mul_arr[valid]
        true_b     = true_bin_arr[valid]
        n          = int(valid.sum())

        bin_acc  = accuracy_score(true_b, bin_preds)
        bin_f1   = f1_score(true_b, bin_preds, **KW)
        bin_prec = precision_score(true_b, bin_preds, **KW)
        bin_rec  = recall_score(true_b, bin_preds, **KW)
        bin_mcc  = matthews_corrcoef(true_b, bin_preds)
        bin_auc  = roc_auc_score(true_b, bin_scores)

        mul_acc  = accuracy_score(true_m, mul_preds)
        mul_f1   = f1_score(true_m, mul_preds, **KW)
        mul_prec = precision_score(true_m, mul_preds, **KW)
        mul_rec  = recall_score(true_m, mul_preds, **KW)
        mul_mcc  = matthews_corrcoef(true_m, mul_preds)
        mul_auc  = roc_auc_score(true_b, bin_scores)  # binary detection AUC

        print(f"{name:<24} {n:>5}  "
              f"{bin_acc:>7.4f} {bin_f1:>7.4f} {bin_prec:>8.4f} {bin_rec:>7.4f} "
              f"{bin_mcc:>7.4f} {bin_auc:>7.4f}  "
              f"{mul_acc:>7.4f} {mul_f1:>7.4f} {mul_prec:>8.4f} {mul_rec:>7.4f} "
              f"{mul_mcc:>7.4f} {mul_auc:>7.4f}")
        metrics_rows.append({
            "model":    name,    "n":       n,
            "bin_acc":  round(bin_acc,  4), "bin_f1":   round(bin_f1,   4),
            "bin_prec": round(bin_prec, 4), "bin_rec":  round(bin_rec,  4),
            "bin_mcc":  round(bin_mcc,  4), "bin_auc":  round(bin_auc,  4),
            "mul_acc":  round(mul_acc,  4), "mul_f1":   round(mul_f1,   4),
            "mul_prec": round(mul_prec, 4), "mul_rec":  round(mul_rec,  4),
            "mul_mcc":  round(mul_mcc,  4), "mul_auc":  round(mul_auc,  4),
        })

    print("=" * 100)
    print("* LightGBM trained on all flow data (no held-out set) - metrics may be inflated.")
    if not HEIFIP_AVAILABLE:
        print("* Packet CNN metrics are uniform (heiFIP binary not found).")

    metrics_rows = _append_meta_row(metrics_rows)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUT_DIR / "metrics_summary.csv", index=False)

    _save_metrics_table_png(metrics_rows, OUT_DIR / "metrics_table_binary.png",     task="binary")
    _save_metrics_table_png(metrics_rows, OUT_DIR / "metrics_table_multiclass.png", task="multiclass")

    # ── Confusion matrices and per-class reports for both ensemble methods ────
    ensemble_variants = [
        ("Weighted Soft-Vote",  "ens_mul_pred",  "ens_bin_pred",
         "wsv_confusion_binary.png", "wsv_confusion_multiclass.png",
         "wsv_classification_report.txt"),
        ("Max-Confidence Vote", "mcv_mul_pred",   "mcv_bin_pred",
         "mcv_confusion_binary.png",   "mcv_confusion_multiclass.png",
         "mcv_classification_report.txt"),
        ("Min-Confidence Vote", "mincv_mul_pred", "mincv_bin_pred",
         "mincv_confusion_binary.png", "mincv_confusion_multiclass.png",
         "mincv_classification_report.txt"),
    ]

    for (label, mul_col, bin_col,
         bin_fname, mul_fname, report_fname) in ensemble_variants:

        mul_preds = df[mul_col].values
        bin_preds = df[bin_col].values

        for true_labels, pred_labels, names, title, fname in [
            (true_bin_arr, bin_preds,
             ["benign", "malicious"],
             f"{label} - Binary Confusion Matrix",
             bin_fname),
            (true_mul_arr, mul_preds,
             CNN_CLASSES,
             f"{label} - Multi-class Confusion Matrix",
             mul_fname),
        ]:
            n_cls = len(names)
            cm    = confusion_matrix(true_labels, pred_labels, labels=list(range(n_cls)))
            plt.figure(figsize=(max(8, n_cls), max(6, n_cls - 1)))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=names, yticklabels=names)
            plt.title(title)
            plt.ylabel("True")
            plt.xlabel("Predicted")
            plt.tight_layout()
            plt.savefig(OUT_DIR / fname, dpi=150)
            plt.close()
            print(f"Saved {OUT_DIR / fname}")

        report = classification_report(
            true_mul_arr, mul_preds,
            target_names=CNN_CLASSES,
            zero_division=0,
        )
        print(f"\n{label} - Multi-class Classification Report:")
        print(report)
        (OUT_DIR / report_fname).write_text(report)

    # ── MCV winner distribution (which model won most often) ─────────────────
    winner_counts = df["mcv_winner_model"].value_counts()
    print("\nMax-Confidence Vote - winner model distribution:")
    for model, count in winner_counts.items():
        print(f"  {model:<12} {count:>5}  ({100 * count / len(df):.1f}%)")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed / 60:.1f} minutes.")
    print(f"All outputs saved to {OUT_DIR}")


def _append_meta_row(metrics_rows: list[dict]) -> list[dict]:
    """If meta_results.csv exists, append the Meta-LightGBM OOF row."""
    meta_csv = BASE / "models" / "meta" / "meta_results.csv"
    if not meta_csv.exists():
        return metrics_rows
    meta_df = pd.read_csv(meta_csv)
    for task in ("binary", "multiclass"):
        prefix = "bin_" if task == "binary" else "mul_"
        oof = meta_df[(meta_df["task"] == task) & (meta_df["fold"] == "OOF")]
        if oof.empty:
            continue
        oof = oof.iloc[0]
        # Find the existing row for "Meta-LightGBM" or append a new one
        for r in metrics_rows:
            if r["model"] == "Meta-LightGBM":
                r[f"{prefix}acc"]  = round(oof["acc"],  4)
                r[f"{prefix}f1"]   = round(oof["f1"],   4)
                r[f"{prefix}prec"] = round(oof["prec"], 4)
                r[f"{prefix}rec"]  = round(oof["rec"],  4)
                r[f"{prefix}mcc"]  = round(oof["mcc"],  4)
                r[f"{prefix}auc"]  = round(oof["auc"],  4)
                break
        else:
            if task == "binary":
                metrics_rows.append({
                    "model":    "Meta-LightGBM", "n": "OOF",
                    "bin_acc":  round(oof["acc"],  4), "bin_f1":   round(oof["f1"],   4),
                    "bin_prec": round(oof["prec"], 4), "bin_rec":  round(oof["rec"],  4),
                    "bin_mcc":  round(oof["mcc"],  4), "bin_auc":  round(oof["auc"],  4),
                    "mul_acc": 0, "mul_f1": 0, "mul_prec": 0,
                    "mul_rec": 0, "mul_mcc": 0, "mul_auc": 0,
                })
    return metrics_rows


def regenerate_tables():
    """
    Rebuild metrics_table_binary.png and metrics_table_multiclass.png from
    the existing raw_results.csv without re-running the full evaluation.
    Run with:  python evaluate_ensemble.py --tables-only
    """
    raw_csv = OUT_DIR / "raw_results.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(f"{raw_csv} not found - run the full evaluation first.")

    df = pd.read_csv(raw_csv)

    def _bin_score(bp_col, conf_col):
        bp = df[bp_col].values; c = df[conf_col].values
        return np.where(bp == 1, c, 1.0 - c)

    if "pkt_grey_bin_score" in df.columns:
        score_map = {
            "Packet_Greyscale":   df["pkt_grey_bin_score"].values,
            "Packet_RGB":         df["pkt_rgb_bin_score"].values,
            "Flow_Greyscale":     df["flw_grey_bin_score"].values,
            "Flow_RGB":           df["flw_rgb_bin_score"].values,
            "Combined_Greyscale": df["comb_grey_bin_score"].values,
            "Combined_RGB":       df["comb_rgb_bin_score"].values,
            "LightGBM *":         df["lgb_bin_score"].values,
            "Weighted Soft-Vote": df["ens_bin_score"].values,
            "Max-Confidence Vote":df["mcv_bin_score"].values,
            **({"Min-Confidence Vote": df["mincv_bin_score"].values}
               if "mincv_bin_score" in df.columns else {}),
        }
    else:
        score_map = {
            "Packet_Greyscale":   _bin_score("pkt_grey_bin_pred",  "pkt_grey_mul_conf"),
            "Packet_RGB":         _bin_score("pkt_rgb_bin_pred",   "pkt_rgb_mul_conf"),
            "Flow_Greyscale":     _bin_score("flw_grey_bin_pred",  "flw_grey_mul_conf"),
            "Flow_RGB":           _bin_score("flw_rgb_bin_pred",   "flw_rgb_mul_conf"),
            "Combined_Greyscale": _bin_score("comb_grey_bin_pred", "comb_grey_mul_conf"),
            "Combined_RGB":       _bin_score("comb_rgb_bin_pred",  "comb_rgb_mul_conf"),
            "LightGBM *":         _bin_score("lgb_bin_pred",       "lgb_mul_conf"),
            "Weighted Soft-Vote": np.mean([
                _bin_score("pkt_grey_bin_pred",  "pkt_grey_mul_conf"),
                _bin_score("pkt_rgb_bin_pred",   "pkt_rgb_mul_conf"),
                _bin_score("flw_grey_bin_pred",  "flw_grey_mul_conf"),
                _bin_score("flw_rgb_bin_pred",   "flw_rgb_mul_conf"),
                _bin_score("comb_grey_bin_pred", "comb_grey_mul_conf"),
                _bin_score("comb_rgb_bin_pred",  "comb_rgb_mul_conf"),
                _bin_score("lgb_bin_pred",       "lgb_mul_conf"),
            ], axis=0),
            "Max-Confidence Vote":_bin_score("mcv_bin_pred", "mcv_winner_conf"),
        }

    model_cols = [
        ("Packet_Greyscale",    "pkt_grey_mul_pred",  "pkt_grey_bin_pred"),
        ("Packet_RGB",          "pkt_rgb_mul_pred",   "pkt_rgb_bin_pred"),
        ("Flow_Greyscale",      "flw_grey_mul_pred",  "flw_grey_bin_pred"),
        ("Flow_RGB",            "flw_rgb_mul_pred",   "flw_rgb_bin_pred"),
        ("Combined_Greyscale",  "comb_grey_mul_pred", "comb_grey_bin_pred"),
        ("Combined_RGB",        "comb_rgb_mul_pred",  "comb_rgb_bin_pred"),
        ("LightGBM *",          "lgb_mul_pred",       "lgb_bin_pred"),
        ("Weighted Soft-Vote",  "ens_mul_pred",   "ens_bin_pred"),
        ("Max-Confidence Vote", "mcv_mul_pred",   "mcv_bin_pred"),
        *([("Min-Confidence Vote", "mincv_mul_pred", "mincv_bin_pred")]
          if "mincv_mul_pred" in df.columns else []),
    ]
    KW = dict(average="macro", zero_division=0)
    y_bin = df["true_binary"].values
    y_mul = df["true_mul"].values

    metrics_rows = []
    for name, mul_col, bin_col in model_cols:
        sc    = score_map[name]
        valid = df[mul_col] >= 0
        mp, bp, sc_v = df.loc[valid, mul_col].values, df.loc[valid, bin_col].values, sc[valid]
        tm, tb, n    = y_mul[valid], y_bin[valid], int(valid.sum())
        metrics_rows.append({
            "model": name, "n": n,
            "bin_acc":  round(accuracy_score(tb, bp), 4),
            "bin_f1":   round(f1_score(tb, bp, **KW), 4),
            "bin_prec": round(precision_score(tb, bp, **KW), 4),
            "bin_rec":  round(recall_score(tb, bp, **KW), 4),
            "bin_mcc":  round(matthews_corrcoef(tb, bp), 4),
            "bin_auc":  round(roc_auc_score(tb, sc_v), 4),
            "mul_acc":  round(accuracy_score(tm, mp), 4),
            "mul_f1":   round(f1_score(tm, mp, **KW), 4),
            "mul_prec": round(precision_score(tm, mp, **KW), 4),
            "mul_rec":  round(recall_score(tm, mp, **KW), 4),
            "mul_mcc":  round(matthews_corrcoef(tm, mp), 4),
            "mul_auc":  round(roc_auc_score(tb, sc_v), 4),
        })

    metrics_rows = _append_meta_row(metrics_rows)
    pd.DataFrame(metrics_rows).to_csv(OUT_DIR / "metrics_summary.csv", index=False)
    _save_metrics_table_png(metrics_rows, OUT_DIR / "metrics_table_binary.png",     task="binary")
    _save_metrics_table_png(metrics_rows, OUT_DIR / "metrics_table_multiclass.png", task="multiclass")


if __name__ == "__main__":
    import sys
    if "--tables-only" in sys.argv:
        regenerate_tables()
    else:
        run_evaluation()
