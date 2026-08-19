# Three-Branch Ensemble for Image-Based Network Intrusion Detection

Code accompanying:

> C. S. Karydis, C. Leka, M. F. Ali, and C. Ntantogian, "Transforming Network Traffic into Images: A Three-Branch Ensemble for Network Intrusion Detection in IoT Networks," 2026.

A network intrusion detection system that fuses three views of the same traffic (packet-level greyscale images, flow-level RGB images and 59 handcrafted statistical flow features) via weighted soft-voting, evaluated on a 16-class IoT attack taxonomy (15 attack families + normal browsing).

## Repository contents

This repository contains the **pipeline code** described in the paper (dataset splitting, image/feature generation, model training, ensemble evaluation and the deployed inference application) plus the **raw attack pcaps and the authors' own benign capture** (`data/raw_pcaps/`, 225MB), so the full pipeline can be reproduced end to end from this repository alone for everything except one file.

Not included here:

- **The CICIDS-2017 benign subset** (534MB, one file). This single file exceeds GitHub's 100MB per-file limit, so it is instead distributed via the full dataset package on Zenodo: https://doi.org/10.5281/zenodo.21926835. It can also be obtained directly from CICIDS-2017's own release.
- **Trained model weights.** These are reproducible by running the training scripts below on the included/linked source data. No pretrained checkpoints are distributed here.

## Repository structure

```
analyze.py                  Deployed Gradio inference application (single-capture triage)
predict.py                  CLI inference / Meta-LightGBM stacking predictions
evaluate_ensemble.py        First-phase (per-branch-independent split) ensemble evaluation
train_branch3.py            LightGBM branch training (first-phase / 5-fold CV protocol)
train_compare.py            CNN architecture-search training (6 configurations)
train_meta.py                Meta-LightGBM stacking (first-phase)
split_flows.py              Multi-flow pcap -> per-flow pcap splitter
process_all_pcaps.sh        Batch packet-image generation via heiFIP
process_flows.sh            Batch flow-image generation via heiFIP

experiments/clean_split/    Unified, leak-free 70/15/15 split pipeline (this paper's headline results)
  build_manifest.py           Pools flows per class, stratified 70/15/15 split (fixed seed)
  build_combined_dirs.py      Builds combined packet+flow image pools (first-phase configs)
  generate_images.py          Packet/flow image generation for the unified split
  generate_images_extra.py    Image generation for the 4 non-deployed CNN configs
  extract_features.py         59-feature statistical extraction (dpkt) for the unified split
  train_cnn.py                DualHead ResNet18 training on the unified split
  train_lightgbm.py           LightGBM training + differential-evolution threshold calibration
  build_meta_features.py      Stacks all 7 base models' probabilities for Meta-LightGBM
  train_meta_unified.py       Meta-LightGBM stacking, trained on val / evaluated on test
  evaluate_clean_ensemble.py  WSV ensemble evaluation on the unified test split
  predict_test_set.py         Per-branch test-set prediction export
  select_strategy_on_val.py   Fusion-strategy and calibration-branch selection on the
                               validation split only (never the test split)
  run_meta_pipeline.sh        End-to-end orchestration script
```

## Modified heiFIP

The flow-level RGB image encoding (Section IV of the paper) required adding an `--rgb` mode to the upstream [heiFIP](https://github.com/stefanDeveloper/heiFIP) C++ tool. That modification lives in a separate fork, kept under heiFIP's own EUPL v1.2 license rather than vendored into this MIT-licensed repository: **https://github.com/chriskarydis/heiFIP** (branch `rgb-flow-mode`).

## Data sources

```
data/raw_pcaps/
  malicious/    15 attack-family captures (0day.pcap, blackEnergy.pcap, mirai.pcap, ...), 225MB total
  benign/       normal_browsing.pcap, the authors' own capture, 244KB
```

- **Attack pcaps (15 classes, included above):** originally released by Rose et al., *913 Malicious Network Traffic PCAPs and Binary Visualisation Images Dataset*, IEEE DataPort, doi:10.21227/pda3-zy88 (CC BY, redistributed here with attribution per IEEE DataPort's Open Access terms). See the citation in the paper for full details.
- **Benign class:** `normal_browsing.pcap` (included above) plus a subset of [CICIDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html) (Canadian Institute for Cybersecurity). CIC's stated policy permits redistribution with citation ("You may redistribute, republish, and mirror our datasets in any form; however, any use or redistribution of the data must include a citation to the dataset and the research paper."), but the file itself is too large for this repository. Get it from the full package on Zenodo (https://doi.org/10.5281/zenodo.21926835) or directly from CICIDS-2017's own release.

After obtaining the CICIDS-2017 file, use `split_flows.py` to split multi-flow captures into per-flow pcaps, then `experiments/clean_split/build_manifest.py` to reproduce the unified 70/15/15 split (fixed seed = 42) used for this paper's headline results.

## Dependencies

Python 3.10+. Main packages (see the paper's Table VII for exact versions used):

```
torch, torchvision       # CNN branches
lightgbm                 # LightGBM branch
imbalanced-learn         # SMOTE
dpkt                     # packet parsing
gradio                   # web application (analyze.py)
numpy, scipy, pandas     # numerics / differential-evolution calibration
Pillow                   # image I/O
```

Image generation additionally requires a built copy of the [modified heiFIP](#modified-heifip) binary, plus `libpcap`/`libpng` as its own build dependencies.

## License

Code in this repository is released under the [MIT License](LICENSE). The heiFIP fork linked above retains heiFIP's own EUPL v1.2 license, as required for a derivative work of an EUPL-licensed project.
