#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "=== [1/9] Waiting for image generation to finish ==="
wait_for_pid() {
  while kill -0 "$1" 2>/dev/null; do sleep 5; done
}
if [ -n "$GEN_PID" ]; then wait_for_pid "$GEN_PID"; fi

echo "=== [2/9] Train Packet_RGB ==="
python3 train_cnn.py --images-root images_packet_rgb --name Packet_RGB --out-dir models/Packet_RGB

echo "=== [3/9] Train Flow_Greyscale ==="
python3 train_cnn.py --images-root images_flow_grey --name Flow_Greyscale --out-dir models/Flow_Greyscale

echo "=== [4/9] Build combined image directories ==="
python3 build_combined_dirs.py

echo "=== [5/9] Train Combined_Greyscale ==="
python3 train_cnn.py --images-root images_combined_grey --name Combined_Greyscale --out-dir models/Combined_Greyscale

echo "=== [6/9] Train Combined_RGB ==="
python3 train_cnn.py --images-root images_combined_rgb --name Combined_RGB --out-dir models/Combined_RGB

echo "=== [7/9] Generate aligned val+test predictions for all 6 CNN configs ==="
declare -A ROOTS=(
  [Packet_Greyscale]=images_packet
  [Packet_RGB]=images_packet_rgb
  [Flow_Greyscale]=images_flow_grey
  [Flow_RGB]=images_flow
  [Combined_Greyscale]=images_combined_grey
  [Combined_RGB]=images_combined_rgb
)
declare -A MODES=(
  [Packet_Greyscale]=packet
  [Packet_RGB]=packet
  [Flow_Greyscale]=flow
  [Flow_RGB]=flow
  [Combined_Greyscale]=packet
  [Combined_RGB]=packet
)
for name in Packet_Greyscale Packet_RGB Flow_Greyscale Flow_RGB Combined_Greyscale Combined_RGB; do
  root=${ROOTS[$name]}
  mode=${MODES[$name]}
  for split in val test; do
    echo "  -> $name / $split"
    python3 predict_test_set.py --mode "$mode" \
      --model-path "models/$name/best_model.pt" \
      --images-root "$root" --split "$split" \
      --out "models/$name/${split}_predictions_aligned.npz"
  done
done

echo "=== [8/9] Build meta-features ==="
python3 build_meta_features.py

echo "=== [9/9] Train + evaluate Meta-LightGBM under unified split ==="
python3 train_meta_unified.py

echo "=== DONE ==="
