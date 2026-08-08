#!/usr/bin/env python3
"""
split_flows.py
Splits a pcap file into per-flow pcaps in a single pass.
Supports TCP, UDP, IP, Ethernet (Ether), and 802.3 (Dot3) frames.
Usage: python3 split_flows.py <input.pcap> <output_dir>
"""

import sys
import os
from collections import defaultdict
from scapy.all import PcapReader, wrpcap, IP, TCP, UDP, Ether
from scapy.layers.l2 import Dot3

def get_flow_key(pkt):
    """
    Returns a normalized bidirectional flow key for a packet.
    Priority: TCP > UDP > IP only > Ethernet > Dot3 (802.3)
    """
    if IP in pkt and TCP in pkt:
        src = (pkt[IP].src, pkt[TCP].sport)
        dst = (pkt[IP].dst, pkt[TCP].dport)
        proto = "tcp"
    elif IP in pkt and UDP in pkt:
        src = (pkt[IP].src, pkt[UDP].sport)
        dst = (pkt[IP].dst, pkt[UDP].dport)
        proto = "udp"
    elif IP in pkt:
        src = (pkt[IP].src, 0)
        dst = (pkt[IP].dst, 0)
        proto = "ip"
    elif Ether in pkt:
        src = (pkt[Ether].src, 0)
        dst = (pkt[Ether].dst, 0)
        proto = "ether"
    elif Dot3 in pkt:
        # IEEE 802.3 frames - use MAC addresses as flow key
        src = (pkt[Dot3].src, 0)
        dst = (pkt[Dot3].dst, 0)
        proto = "dot3"
    else:
        return None, None

    # Normalize direction to avoid duplicate bidirectional flows
    key = (proto, tuple(sorted([src, dst])))
    return key, proto

def flow_key_to_name(key):
    """Convert a flow key to a filename-safe string."""
    proto, endpoints = key
    ep1, ep2 = endpoints
    ep1_str = f"{str(ep1[0]).replace(':', '-')}_{ep1[1]}"
    ep2_str = f"{str(ep2[0]).replace(':', '-')}_{ep2[1]}"
    return f"{proto}_{ep1_str}_{ep2_str}"

def split_flows(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    flows = defaultdict(list)
    total = 0
    skipped = 0
    proto_counts = defaultdict(int)

    print(f"  [i] Reading: {input_file}", flush=True)

    # Single pass through the pcap
    with PcapReader(input_file) as reader:
        for pkt in reader:
            total += 1
            if total % 100000 == 0:
                print(f"  [i] Processed {total} packets, {len(flows)} flows so far...", flush=True)

            key, proto = get_flow_key(pkt)
            if key is None:
                skipped += 1
                continue

            proto_counts[proto] += 1
            flows[key].append(pkt)

    print(f"  [i] Total packets: {total}", flush=True)
    print(f"  [i] Skipped (no layer info): {skipped}", flush=True)
    print(f"  [i] Protocol breakdown: {dict(proto_counts)}", flush=True)
    print(f"  [i] Unique flows: {len(flows)}", flush=True)

    if len(flows) == 0:
        print(f"  [!] No flows found!", flush=True)
        return 0

    # Write each flow to its own pcap
    written = 0
    for key, pkts in flows.items():
        name = flow_key_to_name(key) + ".pcap"
        out_path = os.path.join(output_dir, name)
        wrpcap(out_path, pkts)
        written += 1
        if written % 500 == 0:
            print(f"  [i] Written {written}/{len(flows)} flow pcaps...", flush=True)

    print(f"  [i] Done - {written} flow pcaps saved to {output_dir}", flush=True)
    return written

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 split_flows.py <input.pcap> <output_dir>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        sys.exit(1)

    count = split_flows(input_file, output_dir)
    sys.exit(0 if count > 0 else 1)