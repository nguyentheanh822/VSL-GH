# VSL-GH: A Sentence-Level Vietnamese Sign Language Dataset for General Health Examination

Official PyTorch implementation and reproducible benchmark suite for the paper:

> **VSL-GH: A Sentence-Level Vietnamese Sign Language Dataset for General Health Examination**  
> *Nguyễn Thế Anh et al., The Saigon International University (SIU) & Centre for Research and Education of the Deaf and Hard of Hearing (CED)*

---

## Overview

**VSL-GH** is a continuous, sentence-level, multi-view Vietnamese Sign Language (VSL) dataset dedicated to the **Ho Chi Minh City (HCMC)** regional variant in the **general health examination** domain. It supports end-to-end Sign Language Translation (SLT) with temporally aligned frame-level gloss annotations and 3D MediaPipe Holistic keypoints.

### Dataset Summary (Table V)

| Attribute | Train (S01–S04) | Val (S05) | Test (S06) | Total |
| :--- | :---: | :---: | :---: | :---: |
| Sign language videos (Frontal + Lateral) | 7,200 | 600 | 600 | **8,400** |
| Frontal / Lateral clips | 3,600 / 3,600 | 300 / 300 | 300 / 300 | **4,200 / 4,200** |
| Unique sentences | 300 | 300 | 300 | **300** |
| Independent signers | 4 | 1 | 1 | **6** |
| Avg. frames / video (30 FPS, 1080×1080) | — | — | — | **107.9** (55–151) |
| Avg. words / sentence | — | — | — | **6.2** (3–13) |
| Avg. glosses / sentence | — | — | — | **3.91** (2–8) |
| 3D Keypoints per frame (Pose / Face / Hands) | — | — | — | **137** (25 / 70 / 42) |

---

## Repository Structure

```text
VSL-GH/
├── configs/
│   ├── table6_architecture/   # TCN, Conformer, Mamba, Transformer, ST-GCN
│   ├── table7a_supervision/   # Supervised Frame CE, Gloss-Free, CTC+Sup, CTC Only
│   ├── table7b_ablation/      # w/o Velocity, w/o Face, Multi-view (Front + Side)
│   └── table8_loso/           # 6-Fold Leave-One-Subject-Out (S01–S06)
├── data/
│   ├── dataset.json           # 4,200 entries with frame-level gloss boundaries
│   ├── dataset_loso_s01..06.json
│   ├── gloss_vocab.txt        # Full 300-sentence gloss vocabulary (369 glosses + 5 special tokens)
│   ├── trans_vocab.txt        # Vietnamese translation vocabulary
│   └── splits/                # Signer-independent split lists
├── scripts/
│   └── run_benchmark.py       # Multi-seed training & evaluation runner
├── source/
│   ├── data/dataset.py        # VSLDataset with online augmentation & speed-aligned frame labels
│   ├── models/                # TCN, Conformer, Mamba, Transformer, ST-GCN encoders + Transformer decoder
│   ├── utils/                 # Vocab & evaluation metrics (BLEU-4, WER, chrF, ROUGE-L)
│   ├── train.py               # Training entry point
│   ├── evaluate.py            # Beam-search decoding & evaluation entry point
│   └── extract_keypoints.py   # MediaPipe Holistic 3D keypoint extraction script
├── requirements.txt
└── LICENSE
```

---

## Installation

```bash
git clone https://github.com/nguyentheanh822/VSL-GH.git
cd VSL-GH
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** Install the PyTorch/Torchaudio build matching your CUDA version if needed (tested with Python 3.12, PyTorch 2.5.1 + CUDA 12.1).

---

## Dataset Download

Due to GitHub file size limits, the extracted 3D keypoints (`8,400` `.npy` files) and raw annotations/videos are hosted externally:

- **Dataset Download Link:** *(Coming soon — link will be added here)*

After downloading, extract the archive into `data/` so that `data/keypoints/` contains all `*_F.npy` and `*_S.npy` files (see [data/README.md](data/README.md) for the exact layout).

---

## Reproducing Paper Experiments

All experiments use a shared 6-layer Transformer decoder, 822-dim frontal input features (`411` positional + `411` velocity features), AdamW (`lr=5e-5`), batch size `16`, and 5-way online skeleton augmentation across 5 random seeds (`42–46`).

### 1. Run a Single Model (Train + Evaluate)

```bash
# Train TCN baseline (seed 42)
python source/train.py --config configs/table6_architecture/tcn.yaml --seed 42

# Evaluate best checkpoint on the held-out test signer (S06)
python source/evaluate.py \
    --config configs/table6_architecture/tcn.yaml \
    --checkpoint results/table6/tcn/checkpoint_best.pt \
    --split test
```

### 2. Run Full 5-Seed Benchmark Suites

```bash
# Table VI: Architecture Benchmark (TCN, Conformer, Mamba, Transformer, ST-GCN)
python scripts/run_benchmark.py --suite table6 --gpu 0 --seeds 42 43 44 45 46

# Table VII: Ablation Studies (Gloss Supervision + Input Features & Multi-view)
python scripts/run_benchmark.py --suite table7 --gpu 0 --seeds 42 43 44 45 46

# Table VIII: 6-Fold Leave-One-Subject-Out (LOSO) Cross-Validation
python scripts/run_benchmark.py --suite table8 --gpu 0 --seeds 42 43 44 45 46
```

Results and aggregated 5-seed summaries are written automatically to `results/benchmark/summary.txt` and `summary.json`.

---

## Benchmark Results

### Paper Reported Results (Tables VI–VIII)

| Table / Experiment | Configuration | BLEU-4 ↑ | WER (%) ↓ | chrF ↑ | ROUGE-L ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Table VI** | Transformer | 70.65 ± 2.05 | 28.84 ± 1.95 | 74.42 ± 0.99 | 79.38 ± 1.12 |
| **Table VI** | ST-GCN + Transformer | 71.75 ± 0.98 | 27.91 ± 0.89 | 75.58 ± 1.35 | 80.81 ± 0.94 |
| **Table VI** | Mamba | 79.41 ± 2.14 | 20.76 ± 1.68 | 81.60 ± 2.12 | 85.37 ± 1.40 |
| **Table VI** | Conformer | 81.92 ± 0.62 | 17.78 ± 0.68 | **83.99 ± 0.57** | 87.43 ± 0.42 |
| **Table VI** | **TCN** (12.3M params, 150 FPS) | **82.60 ± 0.31** | **17.11 ± 0.65** | 83.95 ± 0.27 | **87.49 ± 0.44** |
| **Table VIIa** | CTC Only (`ctc=0.5, fce=0.0`) | 67.85 ± 2.03 | 29.53 ± 3.25 | 71.66 ± 2.61 | 77.43 ± 1.70 |
| **Table VIIa** | Gloss-Free (`0.0 / 0.0`) | 79.24 ± 0.96 | 21.11 ± 1.42 | 81.41 ± 0.91 | 84.92 ± 1.17 |
| **Table VIIa** | CTC + Supervised (`0.3 / 0.3`) | 78.92 ± 2.44 | 20.04 ± 2.69 | 80.64 ± 2.28 | 84.86 ± 1.86 |
| **Table VIIa** | **Supervised Frame CE (`0.0 / 0.3`)** | **82.60 ± 0.31** | **17.11 ± 0.65** | **83.95 ± 0.27** | **87.49 ± 0.44** |
| **Table VIIb** | w/o Face (Pose & Hands only) | 76.69 ± 1.19 | 23.60 ± 1.30 | 78.59 ± 0.86 | 82.52 ± 0.80 |
| **Table VIIb** | w/o Velocity | 81.42 ± 0.47 | 18.50 ± 0.55 | 82.95 ± 0.45 | 86.71 ± 0.35 |
| **Table VIIb** | Multi-view (Front + Side) | 81.32 ± 1.73 | 18.08 ± 1.57 | 82.95 ± 1.01 | 86.80 ± 1.04 |
| **Table VIII** | **LOSO 6-Fold Average (TCN)** | **80.73 ± 1.96** | **18.94 ± 1.71** | **82.29 ± 1.94** | **86.23 ± 1.29** |

### Updated Verification Results (Fixed Full-300 Gloss Vocabulary & Speed-Aligned Frame Labels, 5 Seeds)

This release includes two data-pipeline improvements over early internal scripts: (1) full 300-sentence gloss vocabulary coverage (0% OOV across all 4,200 clips) and (2) temporal scaling of frame-level gloss boundaries during speed augmentation. Re-running the full 5-seed benchmark (`seeds 42–46`) with this release yields:

| Experiment | BLEU-4 ↑ | WER (%) ↓ | chrF ↑ | ROUGE-L ↑ |
| :--- | :---: | :---: | :---: | :---: |
| Conformer (Supervised Frame CE) | 84.96 ± 2.04 | 15.17 ± 1.82 | 86.21 ± 1.85 | 88.64 ± 1.37 |
| Mamba (Supervised Frame CE) | 84.57 ± 1.74 | 14.88 ± 1.38 | 86.34 ± 1.80 | 89.08 ± 1.28 |
| **TCN (Supervised Frame CE — Baseline)** | **84.23 ± 1.65** | **15.71 ± 1.58** | **86.00 ± 1.60** | **88.74 ± 1.22** |
| Transformer (Supervised Frame CE) | 81.09 ± 1.60 | 19.06 ± 1.76 | 83.10 ± 1.79 | 86.23 ± 1.33 |
| ST-GCN + Transformer | 78.17 ± 1.93 | 21.14 ± 2.09 | 80.14 ± 1.57 | 84.72 ± 1.38 |
| TCN — Gloss-Free | 81.99 ± 1.68 | 18.00 ± 1.85 | 83.62 ± 1.44 | 86.63 ± 1.19 |
| TCN — CTC + Supervised | 81.77 ± 2.83 | 18.09 ± 2.75 | 83.48 ± 2.03 | 86.68 ± 1.84 |
| TCN — CTC Only | 62.80 ± 3.93 | 35.70 ± 4.09 | 65.83 ± 3.47 | 73.06 ± 3.01 |
| TCN — w/o Velocity | 83.73 ± 0.78 | 16.06 ± 1.08 | 85.27 ± 0.83 | 88.13 ± 0.70 |
| TCN — Multi-view (Front + Side) | 83.33 ± 1.86 | 16.10 ± 1.87 | 85.44 ± 1.66 | 88.52 ± 1.20 |
| TCN — w/o Face | 81.26 ± 0.77 | 18.72 ± 1.02 | 83.27 ± 0.79 | 86.52 ± 0.76 |

---

## Citation

If you find VSL-GH or this codebase useful in your research, please cite:

```bibtex
@article{nguyen2026vslgh,
  title   = {VSL-GH: A Sentence-Level Vietnamese Sign Language Dataset for General Health Examination},
  author  = {Nguyen, The Anh and others},
  year    = {2026},
  url     = {https://github.com/nguyentheanh822/VSL-GH}
}
```

---

## License

This codebase is released under the [MIT License](LICENSE). Please refer to the dataset release page for dataset usage terms and ethical guidelines.
