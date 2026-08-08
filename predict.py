#!/usr/bin/env python3
"""
predict.py  -  Live inference on a single flow PCAP file.

Usage:
    python3 predict.py <flow.pcap> [--label <true_class>]

Pipeline:
    1. Generate greyscale + RGB flow images via heiFIP FlowImage
    2. Generate greyscale + RGB packet images via heiFIP PacketImage
    3. Run all 6 CNN base models (pkt_grey, pkt_rgb, flw_grey, flw_rgb,
       comb_grey, comb_rgb) to get binary + multiclass softmax vectors
    4. Run LightGBM on 65 statistical flow features
    5. Assemble 14-feature binary meta-vector and 112-feature multiclass meta-vector
    6. Predict with Meta-LightGBM binary and multiclass classifiers

Outputs (printed to stdout):
    - Per-model binary prediction (benign / malicious)
    - Weighted soft-vote ensemble prediction
    - Meta-LightGBM binary + multiclass prediction with confidence

Requirements: all models in models/ must be trained (run train_meta.py first).
"""

import os
import sys
import json
import subprocess
import tempfile
import warnings
from pathlib import Path
from argparse import ArgumentParser

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import joblib
import dpkt
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE            = Path.home() / "thesis_heifip"
MODEL_PATHS = {
    "pkt_grey":  BASE / "models" / "comparison" / "Packet_Greyscale"  / "best_model.pt",
    "pkt_rgb":   BASE / "models" / "comparison" / "Packet_RGB"        / "best_model.pt",
    "flw_grey":  BASE / "models" / "comparison" / "Flow_Greyscale"    / "best_model.pt",
    "flw_rgb":   BASE / "models" / "comparison" / "Flow_RGB"          / "best_model.pt",
    "comb_grey": BASE / "models" / "comparison" / "Combined_Greyscale"/ "best_model.pt",
    "comb_rgb":  BASE / "models" / "comparison" / "Combined_RGB"      / "best_model.pt",
}
LGB_BIN_PATH    = BASE / "models" / "branch3" / "binary_model.pkl"
LGB_MUL_PATH    = BASE / "models" / "branch3" / "multi_model.pkl"
LGB_LE_PATH     = BASE / "models" / "branch3" / "label_encoder.pkl"
LGB_FEAT_PATH   = BASE / "models" / "branch3" / "feature_cols.json"
LGB_THRESH_PATH = BASE / "models" / "branch3" / "class_thresholds.npy"
META_BIN_PATH   = BASE / "models" / "meta" / "binary_meta.pkl"
META_MUL_PATH   = BASE / "models" / "meta" / "multi_meta.pkl"
META_NAMES_PATH = BASE / "models" / "meta" / "meta_feature_names.json"
HEIFIP          = BASE / "tools" / "heiFIP" / "heiFIP" / "build" / "heiFIP"

# ─── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE        = 32
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HEIFIP_TIMEOUT  = 30
WEIGHTS = {
    "pkt_grey": 0.9868, "pkt_rgb": 0.9834,
    "flw_grey": 0.8303, "flw_rgb": 0.9108,
    "comb_grey": 0.9717, "comb_rgb": 0.9697,
    "lgb": 0.8635,
}
MIN_CONF_THRESHOLD = 0.9
CNN_CLASSES = [
    "0day", "blackEnergy", "distcc_exec_backdoor",
    "hydra_ftp", "hydra_ssh", "java_rmi",
    "mirai", "netbios_ssn", "normal_browsing",
    "replayAttacks", "ruby_drb", "smtp",
    "tomcat", "unreallrcd", "vsftpd", "zeus",
]
NUM_CLASSES = len(CNN_CLASSES)

# ─── CNN Architecture ─────────────────────────────────────────────────────────
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

_UNIFORM_BIN = np.array([0.5, 0.5])
_UNIFORM_MUL = np.ones(NUM_CLASSES) / NUM_CLASSES


def load_cnn(path: Path) -> DualHeadModel:
    m = DualHeadModel(NUM_CLASSES).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m


# ─── CNN Inference ────────────────────────────────────────────────────────────
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


# ─── heiFIP Image Generation ──────────────────────────────────────────────────
def run_heifip(pcap_path: Path, out_dir: Path, mode: str, rgb: bool) -> list[Path]:
    """Run heiFIP in the given mode (PacketImage or FlowImage), return all PNGs."""
    if not HEIFIP.exists():
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pcap_path.stem
    cmd  = [str(HEIFIP), "-i", str(pcap_path), "-o", str(out_dir),
            "-m", mode, "--min-pkts", "1", "--name", stem]
    if rgb:
        cmd.append("--rgb")
    try:
        subprocess.run(cmd, capture_output=True, timeout=HEIFIP_TIMEOUT)
    except Exception:
        pass
    return sorted(out_dir.glob(f"{stem}*.png"))


# ─── Statistical Feature Extraction (mirrors evaluate_ensemble.py exactly) ───
def extract_features(pcap_path: str) -> dict | None:
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
                arp = 1; continue
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
                if dport == 53 or sport == 53:                 dns     = 1
                if dport == 67 or dport == 68:                 dhcp    = 1
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
                if dport in (20, 21) or sport in (20, 21):     ftp     = 1
                if dport in (139, 445) or sport in (139, 445): netbios = 1
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


# ─── LightGBM Inference ───────────────────────────────────────────────────────
def lgb_proba(pcap_path: Path, lgb_bin, lgb_mul, lgb_feats, lgb_thresh,
              lgb_classes, reorder_idx):
    feats = extract_features(str(pcap_path))
    if feats is None:
        return None, None
    df = pd.DataFrame([feats]).reindex(columns=lgb_feats, fill_value=0)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    X = df.values
    bin_p = lgb_bin.predict_proba(X)[0]
    mul_p = lgb_mul.predict_proba(X)[0]
    if lgb_thresh is not None:
        mul_p = mul_p * lgb_thresh
        s = mul_p.sum()
        if s > 0:
            mul_p /= s
    aligned = np.zeros(NUM_CLASSES)
    for cnn_i, lgb_i in enumerate(reorder_idx):
        if lgb_i >= 0:
            aligned[cnn_i] = mul_p[lgb_i]
    s = aligned.sum()
    if s > 0:
        aligned /= s
    return bin_p, aligned


# ─── Pretty printing ──────────────────────────────────────────────────────────
def _bar(p: float, width: int = 20) -> str:
    filled = int(round(p * width))
    return "[" + "█" * filled + "░" * (width - filled) + f"] {p*100:5.1f}%"


def predict(pcap_path: Path, true_label: str | None = None):
    pcap_path = Path(pcap_path).expanduser().resolve()
    if not pcap_path.exists():
        sys.exit(f"[ERROR] File not found: {pcap_path}")

    print(f"\n{'='*65}")
    print(f"  Predicting: {pcap_path.name}")
    if true_label:
        print(f"  True label: {true_label}")
    print(f"{'='*65}\n")

    # ── Load all models ───────────────────────────────────────────────────────
    print("Loading models...")
    cnn_models = {k: load_cnn(p) for k, p in MODEL_PATHS.items()}

    lgb_bin    = joblib.load(LGB_BIN_PATH)
    lgb_mul    = joblib.load(LGB_MUL_PATH)
    lgb_le     = joblib.load(LGB_LE_PATH)
    lgb_feats  = json.load(open(LGB_FEAT_PATH))
    lgb_thresh = np.load(LGB_THRESH_PATH) if LGB_THRESH_PATH.exists() else None
    lgb_classes = list(lgb_le.classes_)
    reorder_idx = [lgb_classes.index(c) if c in lgb_classes else -1
                   for c in CNN_CLASSES]

    meta_bin   = joblib.load(META_BIN_PATH)
    meta_mul   = joblib.load(META_MUL_PATH)
    meta_names = json.load(open(META_NAMES_PATH))

    print("  All models loaded.\n")

    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)

        # ── Generate images via heiFIP ────────────────────────────────────────
        print("Generating images via heiFIP...")
        flw_grey_imgs = run_heifip(pcap_path, tmp / "flw_g", "FlowImage",   rgb=False)
        flw_rgb_imgs  = run_heifip(pcap_path, tmp / "flw_r", "FlowImage",   rgb=True)
        pkt_grey_imgs = run_heifip(pcap_path, tmp / "pkt_g", "PacketImage", rgb=False)
        pkt_rgb_imgs  = run_heifip(pcap_path, tmp / "pkt_r", "PacketImage", rgb=True)

        flw_grey_ok = bool(flw_grey_imgs)
        flw_rgb_ok  = bool(flw_rgb_imgs)
        pkt_grey_ok = bool(pkt_grey_imgs)
        pkt_rgb_ok  = bool(pkt_rgb_imgs)

        print(f"  FlowImage  greyscale : {len(flw_grey_imgs)} image(s)")
        print(f"  FlowImage  RGB       : {len(flw_rgb_imgs)} image(s)")
        print(f"  PacketImage greyscale: {len(pkt_grey_imgs)} image(s)")
        print(f"  PacketImage RGB      : {len(pkt_rgb_imgs)} image(s)\n")

        # ── Run base models ───────────────────────────────────────────────────
        print("Running base models...")

        pkt_grey_bin, pkt_grey_mul = cnn_proba(pkt_grey_imgs, cnn_models["pkt_grey"])
        pkt_rgb_bin,  pkt_rgb_mul  = cnn_proba(pkt_rgb_imgs,  cnn_models["pkt_rgb"])
        flw_grey_bin, flw_grey_mul = cnn_proba(flw_grey_imgs, cnn_models["flw_grey"])
        flw_rgb_bin,  flw_rgb_mul  = cnn_proba(flw_rgb_imgs,  cnn_models["flw_rgb"])
        comb_grey_bin, comb_grey_mul = cnn_proba(
            (flw_grey_imgs or []) + pkt_grey_imgs, cnn_models["comb_grey"])
        comb_rgb_bin,  comb_rgb_mul  = cnn_proba(
            (flw_rgb_imgs or []) + pkt_rgb_imgs,  cnn_models["comb_rgb"])

        lgb_bin_p, lgb_mul_p = lgb_proba(pcap_path, lgb_bin, lgb_mul, lgb_feats,
                                          lgb_thresh, lgb_classes, reorder_idx)
        lgb_ok = lgb_bin_p is not None
        if not lgb_ok:
            lgb_bin_p = np.array([0.5, 0.5])
            lgb_mul_p = np.ones(NUM_CLASSES) / NUM_CLASSES

        # ── Weighted soft-vote ensemble ───────────────────────────────────────
        w = WEIGHTS
        lgb_w = w["lgb"] if lgb_ok else 0.0
        w_tot = (w["pkt_grey"] + w["pkt_rgb"] + w["flw_grey"] + w["flw_rgb"]
                 + w["comb_grey"] + w["comb_rgb"] + lgb_w)
        ens_mul = (
            w["pkt_grey"]  * pkt_grey_mul  + w["pkt_rgb"]   * pkt_rgb_mul  +
            w["flw_grey"]  * flw_grey_mul  + w["flw_rgb"]   * flw_rgb_mul  +
            w["comb_grey"] * comb_grey_mul + w["comb_rgb"]  * comb_rgb_mul
        )
        ens_bin = (
            w["pkt_grey"]  * pkt_grey_bin  + w["pkt_rgb"]   * pkt_rgb_bin  +
            w["flw_grey"]  * flw_grey_bin  + w["flw_rgb"]   * flw_rgb_bin  +
            w["comb_grey"] * comb_grey_bin + w["comb_rgb"]  * comb_rgb_bin
        )
        if lgb_ok:
            ens_mul += w["lgb"] * lgb_mul_p
            ens_bin += w["lgb"] * lgb_bin_p
        ens_mul /= w_tot
        ens_bin /= w_tot

        # ── Min-Confidence Vote ───────────────────────────────────────────────
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
        _wt        = sum(wt for _, _, wt in _qual)
        mincv_mul  = sum(wt * mp for _, mp, wt in _qual) / _wt
        mincv_bin  = sum(wt * bp for bp, _, wt in _qual) / _wt
        n_qual     = len(_qual)

        # ── Assemble meta-features ────────────────────────────────────────────
        bin_vecs = np.concatenate([
            pkt_grey_bin, pkt_rgb_bin, flw_grey_bin, flw_rgb_bin,
            comb_grey_bin, comb_rgb_bin, lgb_bin_p,
        ])
        mul_vecs = np.concatenate([
            pkt_grey_mul, pkt_rgb_mul, flw_grey_mul, flw_rgb_mul,
            comb_grey_mul, comb_rgb_mul, lgb_mul_p,
        ])

        X_bin_meta = pd.DataFrame([bin_vecs], columns=meta_names["binary"])
        X_mul_meta = pd.DataFrame([mul_vecs], columns=meta_names["multiclass"])

        # ── Meta-LightGBM predictions ─────────────────────────────────────────
        meta_bin_proba  = meta_bin.predict_proba(X_bin_meta)[0]   # [P(benign), P(malicious)]
        meta_mul_proba  = meta_mul.predict_proba(X_mul_meta)[0]   # [p_class_0 ... p_class_15]
        meta_bin_pred   = int(meta_bin.predict(X_bin_meta)[0])
        meta_mul_pred   = int(meta_mul.predict(X_mul_meta)[0])

    # ── Results ───────────────────────────────────────────────────────────────
    BINARY = {0: "benign", 1: "malicious"}

    print("\n" + "─" * 65)
    print("  BASE MODEL RESULTS")
    print("─" * 65)

    base_results = [
        ("Packet  Greyscale", pkt_grey_bin,  pkt_grey_mul),
        ("Packet  RGB",       pkt_rgb_bin,   pkt_rgb_mul),
        ("Flow    Greyscale", flw_grey_bin,  flw_grey_mul),
        ("Flow    RGB",       flw_rgb_bin,   flw_rgb_mul),
        ("Combined Greyscale",comb_grey_bin, comb_grey_mul),
        ("Combined RGB",      comb_rgb_bin,  comb_rgb_mul),
        ("LightGBM",          lgb_bin_p,     lgb_mul_p),
    ]

    for name, bin_p, mul_p in base_results:
        bp = int(bin_p.argmax())
        mp = int(mul_p.argmax())
        flag = "" if not lgb_ok and name == "LightGBM" else ""
        print(f"  {name:<20}  binary: {BINARY[bp]:<10} "
              f"({bin_p[1]*100:5.1f}% mal)  |  "
              f"class: {CNN_CLASSES[mp]:<28} ({mul_p[mp]*100:5.1f}%){flag}")
    if not lgb_ok:
        print("   (LightGBM used uniform fallback - feature extraction failed)")

    print("\n" + "─" * 65)
    print("  WEIGHTED SOFT-VOTE ENSEMBLE")
    print("─" * 65)
    ens_bp = int(ens_bin.argmax())
    ens_mp = int(ens_mul.argmax())
    print(f"  Binary : {BINARY[ens_bp]}  (P(malicious) = {ens_bin[1]*100:.2f}%)")
    print(f"  Class  : {CNN_CLASSES[ens_mp]}  (confidence = {ens_mul[ens_mp]*100:.2f}%)")
    print(f"\n  Top-3 class probabilities:")
    top3 = np.argsort(ens_mul)[::-1][:3]
    for idx in top3:
        print(f"    {CNN_CLASSES[idx]:<28} {_bar(ens_mul[idx])}")

    # Max-Confidence Vote: the single model with highest multiclass confidence decides
    mcv_winner = max(_all, key=lambda x: x[2])
    mcv_bin  = mcv_winner[0]
    mcv_mul  = mcv_winner[1]
    mcv_bp   = int(mcv_bin.argmax())
    mcv_mp   = int(mcv_mul.argmax())

    print("\n" + "─" * 65)
    print("  MAX-CONFIDENCE VOTE")
    print("─" * 65)
    print(f"  Binary : {BINARY[mcv_bp]}  (P(malicious) = {mcv_bin[1]*100:.2f}%)")
    print(f"  Class  : {CNN_CLASSES[mcv_mp]}  (confidence = {mcv_mul[mcv_mp]*100:.2f}%)")
    print(f"\n  Top-3 class probabilities:")
    top3mcv = np.argsort(mcv_mul)[::-1][:3]
    for idx in top3mcv:
        print(f"    {CNN_CLASSES[idx]:<28} {_bar(mcv_mul[idx])}")

    print("\n" + "─" * 65)
    print(f"  MIN-CONFIDENCE VOTE  (threshold={MIN_CONF_THRESHOLD:.0%}, "
          f"{n_qual}/{len(_all)} models qualified)")
    print("─" * 65)
    mincv_bp = int(mincv_bin.argmax())
    mincv_mp = int(mincv_mul.argmax())
    print(f"  Binary : {BINARY[mincv_bp]}  (P(malicious) = {mincv_bin[1]*100:.2f}%)")
    print(f"  Class  : {CNN_CLASSES[mincv_mp]}  (confidence = {mincv_mul[mincv_mp]*100:.2f}%)")
    print(f"\n  Top-3 class probabilities:")
    top3m2 = np.argsort(mincv_mul)[::-1][:3]
    for idx in top3m2:
        print(f"    {CNN_CLASSES[idx]:<28} {_bar(mincv_mul[idx])}")

    print("\n" + "─" * 65)
    print("  META-LIGHTGBM (STACKING)")
    print("─" * 65)
    print(f"  Binary : {BINARY[meta_bin_pred]}  (P(malicious) = {meta_bin_proba[1]*100:.2f}%)")
    print(f"  Class  : {CNN_CLASSES[meta_mul_pred]}  "
          f"(confidence = {meta_mul_proba[meta_mul_pred]*100:.2f}%)")
    print(f"\n  Top-3 class probabilities:")
    top3m = np.argsort(meta_mul_proba)[::-1][:3]
    for idx in top3m:
        print(f"    {CNN_CLASSES[idx]:<28} {_bar(meta_mul_proba[idx])}")

    if true_label is not None:
        true_bin = 0 if true_label == "normal_browsing" else 1
        true_idx = CNN_CLASSES.index(true_label) if true_label in CNN_CLASSES else -1
        print("\n" + "─" * 65)
        print("  CORRECTNESS CHECK")
        print("─" * 65)
        ens_bin_ok   = "✓" if ens_bp == true_bin else "✗"
        ens_mul_ok   = "✓" if ens_mp == true_idx else "✗"
        mcv_bin_ok   = "✓" if mcv_bp == true_bin else "✗"
        mcv_mul_ok   = "✓" if mcv_mp == true_idx else "✗"
        mincv_bin_ok = "✓" if mincv_bp == true_bin else "✗"
        mincv_mul_ok = "✓" if mincv_mp == true_idx else "✗"
        meta_bin_ok  = "✓" if meta_bin_pred == true_bin else "✗"
        meta_mul_ok  = "✓" if meta_mul_pred == true_idx else "✗"
        print(f"  WSV    binary  {ens_bin_ok}  ({BINARY[ens_bp]} vs {BINARY[true_bin]})")
        print(f"  WSV    class   {ens_mul_ok}  ({CNN_CLASSES[ens_mp]} vs {true_label})")
        print(f"  MCV    binary  {mcv_bin_ok}  ({BINARY[mcv_bp]} vs {BINARY[true_bin]})")
        print(f"  MCV    class   {mcv_mul_ok}  ({CNN_CLASSES[mcv_mp]} vs {true_label})")
        print(f"  MiCV   binary  {mincv_bin_ok}  ({BINARY[mincv_bp]} vs {BINARY[true_bin]})")
        print(f"  MiCV   class   {mincv_mul_ok}  ({CNN_CLASSES[mincv_mp]} vs {true_label})")
        print(f"  Meta   binary  {meta_bin_ok}  ({BINARY[meta_bin_pred]} vs {BINARY[true_bin]})")
        print(f"  Meta   class   {meta_mul_ok}  ({CNN_CLASSES[meta_mul_pred]} vs {true_label})")

    print("\n" + "=" * 65 + "\n")

    return {
        "binary_pred":      BINARY[meta_bin_pred],
        "binary_prob_mal":  float(meta_bin_proba[1]),
        "class_pred":       CNN_CLASSES[meta_mul_pred],
        "class_prob":       float(meta_mul_proba[meta_mul_pred]),
        "class_probs":      {CNN_CLASSES[i]: float(meta_mul_proba[i])
                             for i in range(NUM_CLASSES)},
    }


if __name__ == "__main__":
    parser = ArgumentParser(description="Predict attack type from a flow PCAP file.")
    parser.add_argument("pcap", help="Path to the flow PCAP file")
    parser.add_argument("--label", default=None,
                        help=f"True class label for accuracy check. "
                             f"One of: {', '.join(CNN_CLASSES)}")
    args = parser.parse_args()
    predict(args.pcap, args.label)
