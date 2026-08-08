#!/usr/bin/env python3
"""
heiFIP Traffic Analyzer
Web UI for pcap malware classification using weighted soft-voting ensemble:
  - Branch 1: Packet_Greyscale CNN (PacketImage greyscale)
  - Branch 2: Flow_RGB CNN        (FlowImage RGB)
  - Branch 3: LightGBM             (per-flow statistical features)
Final verdict = weighted average of per-class probabilities (weights = macro F1).
"""

import json
import shutil
import subprocess
import warnings
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import dpkt
import joblib

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

import gradio as gr

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE     = Path.home() / "thesis_heifip"
HEIFIP   = BASE / "tools" / "heiFIP" / "heiFIP" / "build" / "heiFIP"
SPLITTER = BASE / "split_flows.py"

PACKET_MODEL_PATH = BASE / "models" / "comparison" / "Packet_Greyscale" / "best_model.pt"
FLOW_MODEL_PATH   = BASE / "models" / "comparison" / "Flow_RGB"          / "best_model.pt"

LGB_BIN_PATH       = BASE / "models" / "branch3" / "binary_model.pkl"
LGB_MUL_PATH       = BASE / "models" / "branch3" / "multi_model.pkl"
LGB_LE_PATH        = BASE / "models" / "branch3" / "label_encoder.pkl"
LGB_FEAT_PATH      = BASE / "models" / "branch3" / "feature_cols.json"
LGB_THRESH_PATH    = BASE / "models" / "branch3" / "class_thresholds.npy"

REPORTS = BASE / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 32
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensemble weights - macro F1 from training
W_PACKET = 0.9868
W_FLOW   = 0.9108
W_LGB    = 0.8635

CNN_CLASSES = [
    "0day", "blackEnergy", "distcc_exec_backdoor",
    "hydra_ftp", "hydra_ssh", "java_rmi",
    "mirai", "netbios_ssn", "normal_browsing",
    "replayAttacks", "ruby_drb", "smtp",
    "tomcat", "unreallrcd", "vsftpd", "zeus"
]
BENIGN_CLASS = "normal_browsing"
NUM_CLASSES  = len(CNN_CLASSES)
IDX_TO_CLASS = {i: c for i, c in enumerate(CNN_CLASSES)}

# ─────────────────────────────────────────────
# CNN model
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


def load_cnn(path: Path):
    model = DualHeadModel(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


PACKET_MODEL = load_cnn(PACKET_MODEL_PATH)
FLOW_MODEL   = load_cnn(FLOW_MODEL_PATH)
print(f"[✓] Packet CNN loaded: {PACKET_MODEL_PATH}")
print(f"[✓] Flow   CNN loaded: {FLOW_MODEL_PATH}")

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ─────────────────────────────────────────────
# LightGBM models
# ─────────────────────────────────────────────
def load_lgb():
    for p in (LGB_BIN_PATH, LGB_MUL_PATH, LGB_LE_PATH, LGB_FEAT_PATH):
        if not p.exists():
            print(f"[!] LightGBM model not found: {p}")
            return None, None, None, None, None
    bin_model = joblib.load(LGB_BIN_PATH)
    mul_model = joblib.load(LGB_MUL_PATH)
    le        = joblib.load(LGB_LE_PATH)
    with open(LGB_FEAT_PATH) as f:
        feat_cols = json.load(f)
    thresholds = np.load(LGB_THRESH_PATH) if LGB_THRESH_PATH.exists() else None
    print(f"[✓] LightGBM models loaded | thresholds: {'yes' if thresholds is not None else 'no'}")
    return bin_model, mul_model, le, feat_cols, thresholds


LGB_BIN, LGB_MUL, LGB_LE, LGB_FEATS, LGB_THRESH = load_lgb()
LGB_AVAILABLE = LGB_BIN is not None

# ─────────────────────────────────────────────
# Feature extraction - mirrors train_branch3.py exactly
# ─────────────────────────────────────────────
def extract_features_from_flow(pcap_path: str) -> dict | None:
    try:
        f = open(pcap_path, "rb")
        pcap = dpkt.pcap.Reader(f)
    except Exception:
        return None

    sizes, timestamps, iats = [], [], []
    header_lens, ttls, proto_types = [], [], []
    dst_ports, src_ports = [], []

    fin_count = syn_count = rst_count = 0
    psh_count = ack_count = ece_count = cwr_count = 0
    http = https = dns = telnet = smtp = ssh = irc = 0
    tcp  = udp   = dhcp = arp   = icmp = igmp = ipv = 0
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
                icmp = 1
                continue
            if isinstance(ip.data, dpkt.igmp.IGMP):
                igmp = 1
                continue

            if isinstance(ip.data, dpkt.udp.UDP):
                udp_pkt = ip.data
                udp = 1
                header_lens.append(8)
                sport, dport = udp_pkt.sport, udp_pkt.dport
                dst_ports.append(dport)
                src_ports.append(sport)
                if dport == 53 or sport == 53:               dns     = 1
                if dport == 67 or dport == 68:               dhcp    = 1
                if dport in (137,138) or sport in (137,138): netbios = 1

            elif isinstance(ip.data, dpkt.tcp.TCP):
                tcp_pkt = ip.data
                tcp = 1
                header_lens.append(tcp_pkt.off * 4)
                sport, dport = tcp_pkt.sport, tcp_pkt.dport
                dst_ports.append(dport)
                src_ports.append(sport)
                flags = tcp_pkt.flags
                if flags & dpkt.tcp.TH_FIN:  fin_count += 1
                if flags & dpkt.tcp.TH_SYN:  syn_count += 1
                if flags & dpkt.tcp.TH_RST:  rst_count += 1
                if flags & dpkt.tcp.TH_PUSH: psh_count += 1
                if flags & dpkt.tcp.TH_ACK:  ack_count += 1
                if flags & 0x40:             ece_count += 1
                if flags & 0x80:             cwr_count += 1
                if dport == 80   or sport == 80:    http    = 1
                if dport == 443  or sport == 443:   https   = 1
                if dport == 22   or sport == 22:    ssh     = 1
                if dport == 23   or sport == 23:    telnet  = 1
                if dport == 25   or sport == 25:    smtp    = 1
                if dport in (194,6667,6668,6697) or \
                   sport in (194,6667,6668,6697):   irc     = 1
                if dport in (20,21) or sport in (20,21):     ftp     = 1
                if dport in (139,445) or sport in (139,445): netbios = 1
    except Exception:
        pass
    finally:
        f.close()

    if not sizes:
        return None

    sizes_arr = np.array(sizes)
    iats_arr  = np.array(iats) if iats else np.array([0.0])
    duration  = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0

    sq25, sq50, sq75 = np.percentile(sizes_arr, [25, 50, 75])
    iq25, iq50, iq75 = np.percentile(iats_arr,  [25, 50, 75])
    _mode = lambda lst: max(set(lst), key=lst.count) if lst else 0

    return {
        "pkt_count":          count,
        "duration":           duration,
        "rate":               count / (duration + 1e-9),
        "bytes_per_sec":      sizes_arr.sum() / (duration + 1e-9),
        "tot_size":           sizes_arr.sum(),
        "min_size":           sizes_arr.min(),
        "max_size":           sizes_arr.max(),
        "avg_size":           sizes_arr.mean(),
        "std_size":           sizes_arr.std(),
        "var_size":           sizes_arr.var(),
        "size_range":         sizes_arr.max() - sizes_arr.min(),
        "size_q25":           float(sq25),
        "size_q50":           float(sq50),
        "size_q75":           float(sq75),
        "iat_mean":           iats_arr.mean(),
        "iat_std":            iats_arr.std(),
        "iat_min":            iats_arr.min(),
        "iat_max":            iats_arr.max(),
        "iat_var":            iats_arr.var(),
        "iat_q25":            float(iq25),
        "iat_q50":            float(iq50),
        "iat_q75":            float(iq75),
        "header_len_mean":    float(np.mean(header_lens)) if header_lens else 0,
        "ttl_mean":           float(np.mean(ttls))        if ttls        else 0,
        "ttl_std":            float(np.std(ttls))         if ttls        else 0,
        "proto_type_mode":    int(_mode(proto_types)),
        "dst_port_mode":      int(_mode(dst_ports)),
        "src_port_mode":      int(_mode(src_ports)),
        "n_unique_dst_ports": len(set(dst_ports)),
        "n_unique_src_ports": len(set(src_ports)),
        "bytes_per_pkt":      sizes_arr.sum() / (count + 1e-9),
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
        "f_smtp":             smtp,
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

# ─────────────────────────────────────────────
# Pcap conversion - handles multi-interface PCAPs
# ─────────────────────────────────────────────
_SUBPROCESS_TIMEOUT = 120  # seconds per tool call

def convert_pcap(input_path: Path, output_path: Path, tmpdir: Path) -> bool:
    tmp = tmpdir / "tmp_conv.pcapng"
    try:
        result = subprocess.run(["capinfos", str(input_path)],
                                capture_output=True, text=True,
                                timeout=_SUBPROCESS_TIMEOUT)
        encap, ifaces = "", 1
        for line in result.stdout.splitlines():
            if "File encapsulation" in line:
                encap = line.split(":", 1)[1].strip().lower()
            if "Number of interfaces" in line:
                try:
                    ifaces = int(line.strip().split()[-1])
                except ValueError:
                    pass

        if "linux cooked" in encap or ifaces > 1:
            # pcapng intermediate for multi-link-type support, then editcap to
            # fix encapsulation, then a second tshark pass to merge interfaces
            # into a single-interface legacy pcap that heiFIP can read
            tmp2 = tmpdir / "tmp_conv2.pcapng"
            subprocess.run(["tshark", "-r", str(input_path), "-w", str(tmp)],
                           capture_output=True, timeout=_SUBPROCESS_TIMEOUT)
            subprocess.run(["editcap", "-T", "ether", str(tmp), str(tmp2)],
                           capture_output=True, timeout=_SUBPROCESS_TIMEOUT)
            subprocess.run(["tshark", "-r", str(tmp2), "-w", str(output_path),
                            "-F", "pcap"], capture_output=True,
                           timeout=_SUBPROCESS_TIMEOUT)
            tmp2.unlink(missing_ok=True)
        else:
            subprocess.run(["tshark", "-r", str(input_path), "-w", str(output_path),
                            "-F", "pcap"], capture_output=True,
                           timeout=_SUBPROCESS_TIMEOUT)

        tmp.unlink(missing_ok=True)
        return output_path.exists()
    except subprocess.TimeoutExpired:
        print(f"[!] Conversion timed out: {input_path.name}")
        return False
    except Exception as e:
        print(f"[!] Conversion error: {e}")
        return False

# ─────────────────────────────────────────────
# CNN inference - returns averaged multi-class proba vector
# ─────────────────────────────────────────────
def cnn_proba_from_images(image_paths: list, model: nn.Module) -> tuple[np.ndarray, np.ndarray]:
    """Returns (avg_bin_proba [2], avg_mul_proba [NUM_CLASSES]) averaged over all images."""
    tensors = []
    for img_path in image_paths:
        try:
            tensors.append(TRANSFORM(Image.open(img_path).convert("RGB")))
        except Exception:
            continue

    if not tensors:
        return np.array([0.5, 0.5]), np.ones(NUM_CLASSES) / NUM_CLASSES

    batch = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        bin_out, mul_out = model(batch)
    bin_probas = torch.softmax(bin_out, dim=1).cpu().numpy()
    mul_probas = torch.softmax(mul_out, dim=1).cpu().numpy()
    return bin_probas.mean(axis=0), mul_probas.mean(axis=0)

# ─────────────────────────────────────────────
# LightGBM inference - returns averaged multi-class proba vector
# ─────────────────────────────────────────────
def xgb_proba_from_flows(flow_pcaps: list) -> tuple[np.ndarray | None, dict]:
    records = []
    for fp in flow_pcaps:
        feats = extract_features_from_flow(str(fp))
        if feats:
            records.append(feats)

    if not records:
        return None, {"available": False, "reason": "no extractable flows"}

    df = pd.DataFrame(records).reindex(columns=LGB_FEATS, fill_value=0)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    X = df.values

    # Per-flow multi-class probabilities
    mul_proba = LGB_MUL.predict_proba(X)   # (n_flows, n_classes)

    # Apply calibrated thresholds if available
    if LGB_THRESH is not None:
        mul_proba = mul_proba * LGB_THRESH
        mul_proba /= mul_proba.sum(axis=1, keepdims=True)

    # Reorder columns to match CNN_CLASSES order
    xgb_classes  = list(LGB_LE.classes_)
    reorder_idx  = [xgb_classes.index(c) if c in xgb_classes else -1
                    for c in CNN_CLASSES]
    aligned = np.zeros((len(mul_proba), NUM_CLASSES))
    for cnn_i, xgb_i in enumerate(reorder_idx):
        if xgb_i >= 0:
            aligned[:, cnn_i] = mul_proba[:, xgb_i]
    aligned /= aligned.sum(axis=1, keepdims=True) + 1e-9

    avg_proba = aligned.mean(axis=0)

    # Per-flow binary predictions for reporting
    bin_preds = LGB_BIN.predict(X)
    mul_names = LGB_LE.inverse_transform(LGB_MUL.predict(X))
    mal_count = int(bin_preds.sum())
    total     = len(bin_preds)
    attack_counts = Counter(
        name for name, is_mal in zip(mul_names, bin_preds)
        if is_mal == 1 and name != BENIGN_CLASS
    )
    stats = {
        "available":   True,
        "total_flows": total,
        "mal_count":   mal_count,
        "ben_count":   total - mal_count,
        "mal_ratio":   round(mal_count / total, 4),
        "top_attacks": attack_counts.most_common(5),
    }
    return avg_proba, stats

# ─────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────
def run_pipeline(pcap_file_path: str, progress=gr.Progress()) -> tuple:
    if not pcap_file_path:
        return "No file uploaded.", "", None

    pcap_path = Path(pcap_file_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workdir   = BASE / "data" / "tmp_analyze" / timestamp
    workdir.mkdir(parents=True, exist_ok=True)

    log_lines = []
    def log(msg):
        log_lines.append(msg)
        print(msg)

    try:
        # ── Step 1: Convert pcap ──────────────────
        progress(0.05, desc="Converting pcap...")
        log(f"[1/6] Converting pcap: {pcap_path.name}")
        conv_path = workdir / "converted.pcap"
        if not convert_pcap(pcap_path, conv_path, workdir):
            log("[!] Conversion failed, using original.")
            conv_path = pcap_path

        # ── Step 2: Split flows ───────────────────
        progress(0.12, desc="Splitting flows...")
        log("[2/6] Splitting into flows...")
        flow_dir = workdir / "flows"
        flow_dir.mkdir()
        subprocess.run(["python3", str(SPLITTER), str(conv_path), str(flow_dir)],
                       capture_output=True, timeout=300)
        flow_pcaps = list(flow_dir.glob("*.pcap"))
        log(f"      {len(flow_pcaps)} flows extracted")

        # ── Step 3: LightGBM inference ─────────────
        progress(0.25, desc="LightGBM flow analysis...")
        log("[3/6] Branch 3 - LightGBM flow feature analysis...")
        if LGB_AVAILABLE:
            lgb_proba, lgb_stats = xgb_proba_from_flows(flow_pcaps)
            if lgb_stats["available"]:
                log(f"      {lgb_stats['total_flows']} flows | "
                    f"malicious: {lgb_stats['mal_count']} ({lgb_stats['mal_ratio']*100:.1f}%)")
            else:
                log(f"      [!] Skipped: {lgb_stats['reason']}")
                lgb_proba = None
        else:
            lgb_proba, lgb_stats = None, {"available": False, "reason": "models not loaded"}
            log("      [!] LightGBM not available.")

        # ── Step 4: Generate greyscale packet images ──
        progress(0.40, desc="Generating packet images...")
        log("[4/6] Generating PacketImage (greyscale) for Branch 1...")
        pkt_img_dir = workdir / "packet_images"
        pkt_img_dir.mkdir()
        subprocess.run([
            str(HEIFIP), "-i", str(conv_path), "-o", str(pkt_img_dir),
            "-m", "PacketImage", "--min-pkts", "1", "--name", "packet"
        ], capture_output=True, timeout=_SUBPROCESS_TIMEOUT)
        packet_images = list(pkt_img_dir.glob("*.png"))
        log(f"      {len(packet_images)} packet images generated")

        progress(0.58, desc="Generating flow images...")
        log("[5/6] Generating FlowImage (RGB) for Branch 2...")
        flow_img_dir = workdir / "flow_images"
        flow_img_dir.mkdir()
        for fp in flow_pcaps:
            try:
                subprocess.run([
                    str(HEIFIP), "-i", str(fp), "-o", str(flow_img_dir),
                    "-m", "FlowImage", "--rgb", "--min-pkts", "1", "--name", fp.stem
                ], capture_output=True, timeout=_SUBPROCESS_TIMEOUT)
            except subprocess.TimeoutExpired:
                continue
        flow_images = list(flow_img_dir.glob("*.png"))
        log(f"      {len(flow_images)} flow images generated")

        # ── Step 6: CNN inference ─────────────────────
        progress(0.75, desc="Running CNN inference...")
        log("[6/6] Running CNN inference...")

        packet_bin_proba, packet_mul_proba = cnn_proba_from_images(packet_images, PACKET_MODEL)
        flow_bin_proba,   flow_mul_proba   = cnn_proba_from_images(flow_images,   FLOW_MODEL)

        log(f"      Packet CNN - binary: {('malicious' if packet_bin_proba[1]>0.5 else 'benign').upper()} "
            f"(conf {packet_bin_proba[1]:.3f})")
        log(f"      Flow   CNN - binary: {('malicious' if flow_bin_proba[1]>0.5 else 'benign').upper()} "
            f"(conf {flow_bin_proba[1]:.3f})")

        # ── Weighted soft-vote ensemble ───────────────
        progress(0.90, desc="Aggregating ensemble...")

        # Multi-class ensemble
        w_total = W_PACKET + W_FLOW + (W_LGB if lgb_proba is not None else 0)
        ensemble_mul = (W_PACKET * packet_mul_proba + W_FLOW * flow_mul_proba) / w_total
        if lgb_proba is not None:
            ensemble_mul += (W_LGB * lgb_proba) / w_total

        # Binary ensemble: derived from the multi-class ensemble (P(benign) vs. 1 - P(benign)),
        # not fused independently, so the binary and multi-class verdicts cannot disagree.
        benign_idx = CNN_CLASSES.index(BENIGN_CLASS)
        p_benign = float(ensemble_mul[benign_idx])
        ensemble_bin = np.array([p_benign, 1.0 - p_benign])

        final_class   = IDX_TO_CLASS[int(ensemble_mul.argmax())]
        final_binary  = "malicious" if ensemble_bin[1] > 0.5 else "benign"
        final_bin_conf = float(ensemble_bin[1] if final_binary == "malicious" else ensemble_bin[0])

        # Top-5 predicted attack types from ensemble
        top5_idx = ensemble_mul.argsort()[::-1][:5]
        top5     = [(IDX_TO_CLASS[i], round(float(ensemble_mul[i]), 4)) for i in top5_idx]

        log(f"      Ensemble verdict: {final_binary.upper()} | "
            f"top class: {final_class} ({ensemble_mul.max():.3f})")

        # ── Build summary ─────────────────────────────
        is_mal   = final_binary == "malicious"
        verdict_icon  = "🔴  MALICIOUS" if is_mal else "🟢  BENIGN"
        conf_bar_len  = int(final_bin_conf * 24)
        conf_bar      = "▓" * conf_bar_len + "░" * (24 - conf_bar_len)

        def branch_verdict(bin_p):
            return ("🔴 MALICIOUS" if bin_p[1] > 0.5 else "🟢 BENIGN") + f"  ({max(bin_p):.1%})"

        lines = [
            "╔══════════════════════════════════════════════════╗",
            "║         TRAFFIC ANALYSIS REPORT              ║",
            "╚══════════════════════════════════════════════════╝",
            f"  FILE    {pcap_path.name}",
            f"  DATE    {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}",
            "",
            "┌─ ENSEMBLE VERDICT ────────────────────────────────────┐",
            f"│  {verdict_icon:<48}│",
            f"│  Confidence  [{conf_bar}]  {final_bin_conf*100:5.1f}%  │",
            f"│  Top class   {final_class:<38}│",
            "└─────────────────────────────────────────────────────────┘",
            "",
            "  CLASS PROBABILITY BREAKDOWN",
            "  " + "─" * 64,
        ]

        max_prob = top5[0][1] if top5 else 1.0
        for cls, prob in top5:
            filled = int((prob / max(max_prob, 1e-9)) * 22)
            bar    = "█" * filled + "░" * (22 - filled)
            marker = " ◀" if cls == final_class else ""
            lines.append(f"  {cls:<26} [{bar}] {prob:.4f}{marker}")

        p1_lbl = IDX_TO_CLASS[int(packet_mul_proba.argmax())]
        p2_lbl = IDX_TO_CLASS[int(flow_mul_proba.argmax())]

        lines += [
            "",
            "  BRANCH DETAILS",
            "  " + "─" * 64,
            f"  ① Packet CNN   {branch_verdict(packet_bin_proba)}",
            f"     images={len(packet_images)}   top={p1_lbl} ({packet_mul_proba.max():.3f})",
            "",
            f"  ② Flow CNN     {branch_verdict(flow_bin_proba)}",
            f"     images={len(flow_images)}   top={p2_lbl} ({flow_mul_proba.max():.3f})",
            "",
            "  ③ LightGBM",
        ]

        if lgb_stats.get("available"):
            mal_pct = lgb_stats["mal_ratio"] * 100
            xgb_bar_len = int(lgb_stats["mal_ratio"] * 22)
            xgb_bar = "█" * xgb_bar_len + "░" * (22 - xgb_bar_len)
            xgb_icon = "🔴" if lgb_stats["mal_ratio"] > 0.5 else "🟢"
            lines += [
                f"     flows={lgb_stats['total_flows']}   "
                f"malicious={lgb_stats['mal_count']} ({mal_pct:.1f}%)",
                f"     {xgb_icon} [{xgb_bar}]",
            ]
            if lgb_stats["top_attacks"]:
                lines.append("     detected attacks:")
                for attack, cnt in lgb_stats["top_attacks"]:
                    pct = cnt / max(lgb_stats["mal_count"], 1) * 100
                    lines.append(f"       · {attack:<28} ×{cnt}  ({pct:.0f}%)")
            else:
                lines.append("     no malicious flows detected")
        else:
            lines.append(f"     unavailable - {lgb_stats.get('reason', 'unknown')}")

        lines += ["", "  " + "─" * 64]
        summary = "\n".join(lines)

        # ── Save JSON report ──────────────────────────
        report = {
            "file":      pcap_path.name,
            "timestamp": datetime.now().isoformat(),
            "verdict":   final_binary,
            "confidence": round(final_bin_conf, 4),
            "predicted_class": final_class,
            "top5_classes": [{"class": c, "prob": p} for c, p in top5],
            "ensemble_weights": {"packet": W_PACKET, "flow": W_FLOW, "xgb": W_LGB},
            "branch_packet": {
                "model":       "Packet_Greyscale",
                "n_images":    len(packet_images),
                "binary_conf": round(float(packet_bin_proba[1]), 4),
                "top_class":   IDX_TO_CLASS[int(packet_mul_proba.argmax())],
                "top_conf":    round(float(packet_mul_proba.max()), 4),
            },
            "branch_flow": {
                "model":       "Flow_RGB",
                "n_images":    len(flow_images),
                "binary_conf": round(float(flow_bin_proba[1]), 4),
                "top_class":   IDX_TO_CLASS[int(flow_mul_proba.argmax())],
                "top_conf":    round(float(flow_mul_proba.max()), 4),
            },
            "branch_lgb": lgb_stats,
        }
        report_path = REPORTS / f"report_{timestamp}_{pcap_path.stem}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        log(f"[✓] Report saved to {report_path}")

        progress(1.0, desc="Done.")
        log("[✓] Analysis complete.")
        return summary, "\n".join(log_lines), str(report_path)

    except Exception as e:
        import traceback
        return f"Pipeline error: {e}\n{traceback.format_exc()}", "\n".join(log_lines), None
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary:   #0a0e1a;
    --bg-card:      #111827;
    --bg-card2:     #1a2234;
    --accent:       #3b82f6;
    --accent-glow:  rgba(59,130,246,0.25);
    --accent2:      #06b6d4;
    --danger:       #ef4444;
    --danger-glow:  rgba(239,68,68,0.2);
    --success:      #22c55e;
    --success-glow: rgba(34,197,94,0.2);
    --text-primary: #f1f5f9;
    --text-muted:   #64748b;
    --border:       rgba(59,130,246,0.15);
}

/* ── Global ─────────────────────────────── */
body, .gradio-container {
    background: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}

/* Animated grid background */
.gradio-container::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(59,130,246,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(59,130,246,0.04) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Header ─────────────────────────────── */
#heifip-header {
    text-align: center;
    padding: 40px 20px 20px;
    position: relative;
}
#heifip-header h1 {
    font-size: 2.4em;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #06b6d4, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    margin-bottom: 8px;
}
#heifip-header .subtitle {
    color: var(--text-muted);
    font-size: 0.92em;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Cards ──────────────────────────────── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    transition: border-color 0.3s, box-shadow 0.3s;
}
.card:hover {
    border-color: rgba(59,130,246,0.35);
    box-shadow: 0 0 24px var(--accent-glow);
}

/* Branch badges */
.branch-pills {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    justify-content: center;
    margin-top: 14px;
}
.branch-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.78em;
    font-family: 'JetBrains Mono', monospace;
    color: #94a3b8;
    transition: all 0.2s;
}
.branch-pill:hover {
    border-color: var(--accent);
    color: var(--text-primary);
    box-shadow: 0 0 10px var(--accent-glow);
}
.branch-pill .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 6px var(--accent);
    animation: pulse-dot 2s ease-in-out infinite;
}
.branch-pill:nth-child(2) .dot { background: var(--accent2); box-shadow: 0 0 6px var(--accent2); }
.branch-pill:nth-child(3) .dot { background: #a78bfa; box-shadow: 0 0 6px #a78bfa; }

@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.75); }
}

/* ── Upload zone ────────────────────────── */
#upload-col .wrap {
    border: 2px dashed var(--border) !important;
    border-radius: 10px !important;
    background: var(--bg-card2) !important;
    transition: border-color 0.25s, background 0.25s !important;
    min-height: 110px !important;
}
#upload-col .wrap:hover {
    border-color: var(--accent) !important;
    background: rgba(59,130,246,0.06) !important;
}
#upload-col .wrap svg { color: var(--accent) !important; }

/* ── Analyze button ─────────────────────── */
#analyze-btn {
    width: 100%;
    margin-top: 12px;
    background: linear-gradient(135deg, #2563eb, #0891b2) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 1em !important;
    letter-spacing: 0.04em !important;
    padding: 14px !important;
    cursor: pointer !important;
    transition: opacity 0.2s, box-shadow 0.2s, transform 0.15s !important;
    box-shadow: 0 0 18px rgba(37,99,235,0.4) !important;
    position: relative;
    overflow: hidden;
}
#analyze-btn::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, transparent 30%, rgba(255,255,255,0.12) 50%, transparent 70%);
    transform: translateX(-100%);
    transition: transform 0.5s;
}
#analyze-btn:hover {
    opacity: 0.92 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 0 28px rgba(37,99,235,0.6) !important;
}
#analyze-btn:hover::after { transform: translateX(100%); }
#analyze-btn:active { transform: translateY(0) !important; }

/* ── Summary output ─────────────────────── */
#summary_out {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    background: var(--bg-card) !important;
}
#summary_out textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82em !important;
    line-height: 1.65 !important;
    color: #cbd5e1 !important;
    background: transparent !important;
    caret-color: var(--accent);
}

/* ── Log output ─────────────────────────── */
#log-section {
    margin-top: 8px;
}
#log_out {
    border: 1px solid rgba(100,116,139,0.2) !important;
    border-radius: 10px !important;
    background: #0d1117 !important;
}
#log_out textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.74em !important;
    color: #475569 !important;
    background: transparent !important;
    line-height: 1.7 !important;
}

/* ── Report download ────────────────────── */
#report-section .file-preview {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* ── Labels ─────────────────────────────── */
label span, .label-wrap span {
    color: #64748b !important;
    font-size: 0.78em !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* ── Scanning animation on button click ─── */
@keyframes scan-line {
    0%   { top: 0; opacity: 0.6; }
    100% { top: 100%; opacity: 0; }
}
.scanning::before {
    content: '';
    position: absolute;
    left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    animation: scan-line 1.2s linear infinite;
    pointer-events: none;
}

/* ── Scrollbar ──────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(59,130,246,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(59,130,246,0.6); }
"""

js = """
function addScanEffect() {
    const btn = document.getElementById('analyze-btn');
    if (!btn) return;
    btn.addEventListener('click', () => {
        btn.classList.add('scanning');
        btn.textContent = '⟳  Analyzing...';
        setTimeout(() => {
            btn.classList.remove('scanning');
            btn.textContent = '⚡  Analyze';
        }, 8000);
    });
}
"""

with gr.Blocks(title="heiFIP Traffic Analyzer", css=css, js=js,
               theme=gr.themes.Base(
                   primary_hue=gr.themes.colors.blue,
                   neutral_hue=gr.themes.colors.slate,
                   font=gr.themes.GoogleFont("Inter"),
               )) as app:

    # ── Header ──────────────────────────────────────────────
    gr.HTML(f"""
    <div id="heifip-header">
        <h1>⚡ heiFIP Traffic Analyzer</h1>
        <p class="subtitle">network intrusion detection &nbsp;·&nbsp; weighted soft-voting ensemble</p>
        <div class="branch-pills">
            <span class="branch-pill"><span class="dot"></span>Branch 1 - Packet CNN &nbsp;<b>w={W_PACKET}</b></span>
            <span class="branch-pill"><span class="dot"></span>Branch 2 - Flow CNN &nbsp;<b>w={W_FLOW}</b></span>
            <span class="branch-pill"><span class="dot"></span>Branch 3 - LightGBM &nbsp;<b>w={W_LGB}</b></span>
        </div>
    </div>
    """)

    # ── Main row ─────────────────────────────────────────────
    with gr.Row(equal_height=False):
        # Left: upload + button
        with gr.Column(scale=1, elem_id="upload-col"):
            pcap_input = gr.File(
                label="Drop or click to upload PCAP",
                file_types=[".pcap"],
                type="filepath",
            )
            analyze_btn = gr.Button(
                "⚡  Analyze",
                variant="primary",
                size="lg",
                elem_id="analyze-btn",
            )
            gr.HTML("""
            <div style="margin-top:16px; padding:14px; background:#111827;
                        border:1px solid rgba(59,130,246,0.12); border-radius:10px;
                        font-size:0.78em; color:#475569; line-height:1.8;
                        font-family:'JetBrains Mono',monospace;">
                <b style="color:#64748b">HOW IT WORKS</b><br>
                1. PCAP converted &amp; normalized<br>
                2. Flows extracted (5-tuple split)<br>
                3. LightGBM scores each flow<br>
                4. Packet &amp; flow images rendered<br>
                5. Two CNNs run inference<br>
                6. Weighted soft-vote ensemble
            </div>
            """)

        # Right: results
        with gr.Column(scale=2):
            summary_out = gr.Textbox(
                label="Analysis Report",
                lines=30,
                elem_id="summary_out",
                interactive=False,
                placeholder="Results will appear here after analysis...",
            )

    # ── Log row ───────────────────────────────────────────────
    with gr.Row(elem_id="log-section"):
        log_out = gr.Textbox(
            label="Pipeline Log",
            lines=7,
            elem_id="log_out",
            interactive=False,
            placeholder="Step-by-step log...",
        )

    # ── Report download ───────────────────────────────────────
    with gr.Row(elem_id="report-section"):
        report_out = gr.File(label="Download JSON Report", interactive=False)

    analyze_btn.click(
        fn=run_pipeline,
        inputs=[pcap_input],
        outputs=[summary_out, log_out, report_out],
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
