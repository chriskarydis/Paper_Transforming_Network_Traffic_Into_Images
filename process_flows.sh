#!/bin/bash

# ─────────────────────────────────────────────
# heiFIP Flow Image Batch Processing Script v3
# Generates both greyscale and RGB flow images
# ─────────────────────────────────────────────

# Paths
BASE=~/thesis_heifip
RAW=$BASE/data/raw
CONVERTED=$BASE/data/converted
FLOWS=$BASE/data/flows
IMAGES=$BASE/data/images_flow
HEIFIP=$BASE/tools/heiFIP/heiFIP/build/heiFIP
SPLITTER=$BASE/split_flows.py
TMPDIR=$BASE/data/tmp
mkdir -p "$TMPDIR"

# heiFIP parameters
THREADS=2
MODE=FlowImage

# ─────────────────────────────────────────────
# Helper: convert a single pcap robustly
# ─────────────────────────────────────────────
convert_pcap() {
    local INPUT=$1
    local OUTPUT=$2
    local TMP=$TMPDIR/tmp_conv.pcap

    local ENCAP=$(capinfos "$INPUT" | grep "File encapsulation" | awk -F: '{print $2}' | xargs)
    local IFACES=$(capinfos "$INPUT" | grep "Number of interfaces" | head -1 | awk '{print $NF}')

    echo "  [i] Encapsulation: $ENCAP | Interfaces: $IFACES"

    if echo "$ENCAP" | grep -qi "linux cooked"; then
        echo "  [i] Strategy: tshark flatten + editcap SLL→Ethernet"
        tshark -r "$INPUT" -w "$TMP" 2>/dev/null
        editcap -T ether "$TMP" "$OUTPUT"
    elif [ "$IFACES" -gt 1 ]; then
        echo "  [i] Strategy: tshark flatten + editcap (multi-interface)"
        tshark -r "$INPUT" -w "$TMP" 2>/dev/null
        editcap -T ether "$TMP" "$OUTPUT"
    else
        echo "  [i] Strategy: tshark flatten only"
        tshark -r "$INPUT" -w "$OUTPUT" -F pcap 2>/dev/null
    fi

    rm -f "$TMP"

    local OUT_ENCAP=$(capinfos "$OUTPUT" | grep "File encapsulation" | awk -F: '{print $2}' | xargs)
    local OUT_IFACES=$(capinfos "$OUTPUT" | grep "Number of interfaces" | head -1 | awk '{print $NF}')
    echo "  [i] Result: Encapsulation=$OUT_ENCAP | Interfaces=$OUT_IFACES"
}

# ─────────────────────────────────────────────
# Helper: process one category folder
# ─────────────────────────────────────────────
process_category() {
    local CATEGORY=$1

    for RGB_MODE in greyscale rgb; do
        local RAW_DIR=$RAW/$CATEGORY
        local CONV_DIR=$CONVERTED/$CATEGORY
        local FLOW_DIR=$FLOWS/$CATEGORY
        local IMG_DIR=$IMAGES/$RGB_MODE/$CATEGORY

        echo ""
        echo "════════════════════════════════════════"
        echo "  Processing category: $CATEGORY [$RGB_MODE]"
        echo "════════════════════════════════════════"

        mkdir -p "$CONV_DIR"
        mkdir -p "$FLOW_DIR"
        mkdir -p "$IMG_DIR"

        # For benign pcaps, flows and images share one output folder each
        if [ "$CATEGORY" = "benign_pcaps" ]; then
            SHARED_FLOW_SUBDIR=$FLOW_DIR/normal_browsing
            SHARED_OUT_DIR=$IMG_DIR/normal_browsing
            mkdir -p "$SHARED_FLOW_SUBDIR"
            mkdir -p "$SHARED_OUT_DIR"
        fi

        for RAW_FILE in "$RAW_DIR"/*.pcap; do
            BASENAME=$(basename "$RAW_FILE" .pcap)
            CONV_FILE=$CONV_DIR/$BASENAME.pcap

            if [ "$CATEGORY" = "benign_pcaps" ]; then
                FLOW_SUBDIR=$SHARED_FLOW_SUBDIR
                OUT_DIR=$SHARED_OUT_DIR
            else
                FLOW_SUBDIR=$FLOW_DIR/$BASENAME
                OUT_DIR=$IMG_DIR/$BASENAME
            fi

            echo ""
            echo "──────────────────────────────────────"
            echo "  [$(date +%H:%M:%S)] $BASENAME [$RGB_MODE]"
            echo "──────────────────────────────────────"

            # Step 1: Convert pcap (skip if already done)
            if [ ! -f "$CONV_FILE" ]; then
                echo "  [1/4] Converting pcap..."
                convert_pcap "$RAW_FILE" "$CONV_FILE"
                if [ $? -ne 0 ] || [ ! -f "$CONV_FILE" ]; then
                    echo "  [!] Conversion failed for $BASENAME, skipping."
                    continue
                fi
            else
                echo "  [1/4] Skipping conversion (already done)"
            fi

            # Step 2: Split into per-flow pcaps (skip if already done for this pcap)
            FLOW_DONE_MARKER="$FLOW_SUBDIR/.split_done_$BASENAME"
            if [ -f "$FLOW_DONE_MARKER" ]; then
                echo "  [2/4] Skipping flow split (already done)"
            else
                # Backfill marker if flow pcaps exist but marker is missing (pre-marker runs)
                EXISTING_FLOWS=$(find "$FLOW_SUBDIR" -name "*.pcap" ! -name ".split_done_*" 2>/dev/null | wc -l)
                if [ "$EXISTING_FLOWS" -gt 0 ]; then
                    echo "  [2/4] Flow pcaps already exist ($EXISTING_FLOWS), backfilling marker and skipping."
                    touch "$FLOW_DONE_MARKER"
                else
                    echo "  [2/4] Splitting into flows (Python single-pass)..."
                    mkdir -p "$FLOW_SUBDIR"
                    python3 "$SPLITTER" "$CONV_FILE" "$FLOW_SUBDIR"
                    [ $? -eq 0 ] && touch "$FLOW_DONE_MARKER"
                fi
            fi

            FLOW_COUNT=$(find "$FLOW_SUBDIR" -name "*.pcap" ! -name ".split_done_*" | wc -l)
            if [ "$FLOW_COUNT" -eq 0 ]; then
                echo "  [!] No flows for $BASENAME, skipping."
                continue
            fi
            echo "  [i] $FLOW_COUNT flow pcaps ready"

            # Step 3: Skip if images already generated for this pcap
            DONE_MARKER="$OUT_DIR/.done_$BASENAME"
            if [ -f "$DONE_MARKER" ]; then
                echo "  [3/4] Skipping image generation (already done)"
                continue
            fi
            # Backfill marker if images exist but marker is missing (pre-marker runs)
            if [ "$CATEGORY" = "benign_pcaps" ]; then
                EXISTING=$(find "$OUT_DIR" -name "${BASENAME}*.png" 2>/dev/null | wc -l)
            else
                EXISTING=$(find "$OUT_DIR" -name "*.png" 2>/dev/null | wc -l)
            fi
            if [ "$EXISTING" -gt 0 ]; then
                echo "  [3/4] Images already exist ($EXISTING), backfilling marker and skipping."
                touch "$DONE_MARKER"
                continue
            fi
            echo "  [3/4] Preparing output folder..."
            mkdir -p "$OUT_DIR"

            # Step 4: Run heiFIP
            echo "  [4/4] Generating $RGB_MODE flow images..."
            RGB_FLAG=""
            [ "$RGB_MODE" = "rgb" ] && RGB_FLAG="--rgb"

            for FLOW_FILE in "$FLOW_SUBDIR"/*.pcap; do
                FLOW_BASENAME=$(basename "$FLOW_FILE" .pcap)
                $HEIFIP \
                    -i "$FLOW_FILE" \
                    -o "$OUT_DIR" \
                    -m $MODE \
                    -t $THREADS \
                    --min-pkts 1 \
                    $RGB_FLAG \
                    --name "$FLOW_BASENAME"
            done

            IMG_COUNT=$(find "$OUT_DIR" -name "*.png" | wc -l)
            if [ "$IMG_COUNT" -eq 0 ]; then
                echo "  [!] Warning: 0 images generated for $BASENAME [$RGB_MODE]"
            else
                touch "$DONE_MARKER"
                echo "  [✓] Done - $IMG_COUNT images saved to $OUT_DIR"
            fi
        done
    done
}

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════╗"
echo "║  heiFIP Flow Image Batch Processing  ║"
echo "╚══════════════════════════════════════╝"
echo "  Started at: $(date)"

process_category "malicious_pcaps"
process_category "benign_pcaps"

rm -rf "$TMPDIR"

echo ""
echo "╔══════════════════════════════════════╗"
echo "║   All processing complete!           ║"
echo "╚══════════════════════════════════════╝"
echo "  Finished at: $(date)"
echo ""
echo "Summary:"
echo "  Greyscale - Malicious: $(find $IMAGES/greyscale/malicious_pcaps -name '*.png' 2>/dev/null | wc -l)"
echo "  Greyscale - Benign:    $(find $IMAGES/greyscale/benign_pcaps -name '*.png' 2>/dev/null | wc -l)"
echo "  RGB       - Malicious: $(find $IMAGES/rgb/malicious_pcaps -name '*.png' 2>/dev/null | wc -l)"
echo "  RGB       - Benign:    $(find $IMAGES/rgb/benign_pcaps -name '*.png' 2>/dev/null | wc -l)"
echo ""