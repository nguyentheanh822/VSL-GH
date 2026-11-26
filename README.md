# VSL-GH: A Sentence-Level Vietnamese Sign Language Dataset for General Health Examination

Official source code and reproducible benchmark suite for the paper:

> **VSL-GH: A Sentence-Level Vietnamese Sign Language Dataset for General Health Examination**  
> **Thế Anh Nguyễn**  
> The Saigon International University (SIU), Ho Chi Minh City, Vietnam  
> Contact: `theannheleven2o3@gmail.com`

---

## Overview

**VSL-GH** is a continuous, sentence-level, multi-view Vietnamese Sign Language (VSL) dataset of the **Ho Chi Minh City (HCMC)** variant dedicated to the **general health examination** domain. Developed in collaboration with the **Centre for Research and Education of the Deaf and Hard of Hearing (CED)** and physicians at **Tan Hung General Hospital** (following Vietnam Ministry of Health Circular 32/2023/TT-BYT), VSL-GH provides synchronized dual-camera recordings, millisecond-precision temporal gloss annotations, Vietnamese sentence translations, and 3D MediaPipe Holistic keypoints.

### Key Highlights
- **8,400 video clips** (4,200 recordings from 6 deaf signers `S01–S06`, captured simultaneously from frontal and 90° lateral DJI Osmo Pocket 3 cameras at 1080×1080 px, 30 fps in a green-screen studio).
- **300 healthcare sentences** (299 distinct; covering 5 examination phases: *Registration*, *Physical Examination*, *Clinical Examination*, *Paraclinical Examination*, and *Conclusion*).
- **Multi-level annotations**: ELAN-annotated millisecond-precision gloss boundaries, Vietnamese sentence translations, and 137 3D landmarks (25 body pose, 70 facial, 42 hand landmarks) in MediaPipe image-normalized coordinates.
- **Signer-independent evaluation protocol**: `S01–S04` (train, 3 repetitions each), `S05` (validation, 1 repetition), and `S06` (test, 1 repetition) with zero signer overlap across splits, plus 6-fold Leave-One-Subject-Out (LOSO) cross-validation.

---

## Dataset Statistics

### Overall Statistics (Table 5)

| Attribute | Train (S01–S04) | Val (S05) | Test (S06) | Total |
| :--- | :---: | :---: | :---: | :---: |
| Sign language videos | 7,200 | 600 | 600 | **8,400** |
| Frontal / Lateral clips | 3,600 / 3,600 | 300 / 300 | 300 / 300 | **4,200 / 4,200** |
| Avg. frames / video (30 fps) | — | — | — | **107.9** (55–151) |
| Sentences | 300 | 300 | 300 | **300** |
| Word tokens / unique | — | — | — | **1,861 / 440** |
| Gloss tokens / unique | — | — | — | **1,190 / 372** |
| Avg. words / sentence | — | — | — | **6.2** (3–13) |
| Avg. glosses / sentence | — | — | — | **3.97** (2–8) |
| Number of signers | 4 | 1 | 1 | **6** |
| 3D keypoints (137 total): Pose / Face / Hands | — | — | — | **25 / 70 / 42** |

### Distribution Across Examination Phases (Table 4)

| Phase | Sent. | Words / sent. | Glosses / sent. | Unique words | Unique glosses | Interrog. (%) | NMM (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Registration | 68 | 6.19 | 4.16 | 157 | 120 | 39.7 | 20.6 |
| Physical Examination | 32 | 6.16 | 3.91 | 76 | 55 | 56.3 | 9.4 |
| Clinical Examination | 130 | 5.63 | 3.66 | 231 | 181 | 22.3 | 47.7 |
| Paraclinical Examination | 40 | 7.47 | 4.42 | 112 | 73 | 40.0 | 0.0 |
| Conclusion | 30 | 7.07 | 4.30 | 99 | 72 | 63.3 | 10.0 |
| **All** | **300** | **6.20** | **3.97** | **440** | **372** | **36.3** | **27.3** |

*Sentence types:* 175 declarative (58.3%), 109 interrogative (36.3%), and 16 exclamatory (5.3%); 82 of 300 sentences (27.3%) carry an explicitly annotated non-manual marker (NMM) note.

---

## Dataset Accessibility

The VSL-GH dataset (videos, extracted 3D keypoints, temporal gloss annotations, Vietnamese text, and metadata) **will be released upon publication** in a dedicated external dataset repository (link and DOI will be added here upon release).

Once downloaded, extract the keypoints and annotations into `data/` following the directory structure documented in [data/README.md](data/README.md).

---

## Repository Structure

```text
VSL-GH/
├── configs/
│   ├── table6_architecture/   # Table 6: TCN, Conformer, Mamba, Transformer, ST-GCN
│   ├── table7a_supervision/   # Table 7a: Supervised Frame CE, Gloss-Free, CTC+Sup, CTC Only
│   ├── table7b_ablation/      # Table 7b: w/o Velocity, w/o Face, Multi-view (Front + Side)
│   ├── table8_loso/           # Table 8: 6-Fold Leave-One-Subject-Out (S01–S06)
│   └── table10_synthetic_scaling/ # Table 10: Synthetic 5× Pre-training + Fine-tuning
├── data/
│   ├── README.md              # Expected layout for external keypoints & annotations
│   ├── dataset.json           # Metadata with frame-aligned gloss boundaries (4,200 clips)
│   ├── dataset_loso_s01..06.json
│   ├── gloss_vocab.txt        # Full gloss vocabulary
│   ├── trans_vocab.txt        # Vietnamese translation vocabulary
│   └── splits/                # Signer-independent split lists (S01–S04 / S05 / S06)
├── scripts/
│   ├── run_benchmark.py       # Multi-seed runner for Tables 6, 7, and 8
│   └── run_synthetic_scaling.py # Synthetic 5× scaling + fine-tuning runner (Table 10)
├── source/
│   ├── data/dataset.py        # Keypoint normalization, velocity computation & augmentation
│   ├── models/                # TCN, Conformer, Mamba, Transformer, ST-GCN + 6-layer Decoder
│   ├── utils/                 # Vocabulary & metrics (BLEU-4, WER, chrF, ROUGE-L)
│   ├── train.py               # Training entry point
│   ├── evaluate.py            # Beam-search decoding (beam=4) & evaluation entry point
│   └── extract_keypoints.py   # MediaPipe Holistic 137-landmark extraction script
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

*(Tested on Linux with Python 3.12, PyTorch 2.5.1 + CUDA 12.1 on a single NVIDIA RTX 4090 GPU.)*

---

## Reproducing Experiments

All encoders share an identical **6-layer Transformer decoder** and use shoulder-normalized frontal-view 3D landmarks concatenated with first-order temporal velocity (`411` positional + `411` velocity = `822` dimensions per frame). Metrics are evaluated on the unseen test signer (`S06`) with beam search (`beam_size = 4`) and reported as **mean ± std across 5 random seeds**.

### Single Experiment (Train & Evaluate)

```bash
# Train TCN baseline (Supervised Frame CE, weight 0.3)
python source/train.py --config configs/table6_architecture/tcn.yaml --seed 42

# Evaluate best checkpoint on the held-out test signer (S06)
python source/evaluate.py \
    --config configs/table6_architecture/tcn.yaml \
    --checkpoint results/table6/tcn/checkpoint_best.pt \
    --split test
```

### Multi-Seed Benchmark Suites (5 Seeds)

```bash
# Table 6: Architecture Benchmark (Transformer, ST-GCN, Mamba, Conformer, TCN)
python scripts/run_benchmark.py --suite table6 --gpu 0 --seeds 42 43 44 45 46

# Table 7: Ablation Studies (Gloss Supervision + Input Features & Multi-view)
python scripts/run_benchmark.py --suite table7 --gpu 0 --seeds 42 43 44 45 46

# Table 8: 6-Fold Leave-One-Subject-Out (LOSO) Cross-Validation
python scripts/run_benchmark.py --suite table8 --gpu 0 --seeds 42 43 44 45 46

# Table 10: Synthetic 5x Data Scaling (offline 5x on S01-S04) + Fine-tuning (beam=4)
python scripts/run_synthetic_scaling.py --gpu 0 --seeds 42 43 44 45 46
```

---

## Benchmark Results

### Table 6 — Architecture Benchmark on VSL-GH (All models use a 6-layer Transformer decoder)

| Model | Params (M) | FPS | BLEU-4 ↑ | WER (%) ↓ | chrF ↑ | ROUGE-L ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Transformer (Vaswani et al., 2017) | 45.2 | 85 | 70.65 ± 2.05 | 28.84 ± 1.95 | 74.42 ± 0.99 | 79.38 ± 1.12 |
| ST-GCN + Transformer (Yan et al., 2018) | 21.5 | 110 | 71.75 ± 0.98 | 27.91 ± 0.89 | 75.58 ± 1.35 | 80.81 ± 0.94 |
| Mamba (Gu & Dao, 2023) | 18.4 | 135 | 79.41 ± 2.14 | 20.76 ± 1.68 | 81.60 ± 2.12 | 85.37 ± 1.40 |
| Conformer (Gulati et al., 2020) | 38.6 | 95 | 81.92 ± 0.62 | 17.78 ± 0.68 | **83.99 ± 0.57** | 87.43 ± 0.42 |
| **TCN (Bai et al., 2018)** | **12.3** | **150** | **82.60 ± 0.31** | **17.11 ± 0.65** | 83.95 ± 0.27 | **87.49 ± 0.44** |

### Table 7 — Ablation Studies on the TCN Encoder

| Configuration | BLEU-4 ↑ | WER (%) ↓ | chrF ↑ | ROUGE-L ↑ |
| :--- | :---: | :---: | :---: | :---: |
| **(a) Gloss Supervision** (`ctc` / `fce` weights) | | | | |
| CTC Only (`0.5 / 0.0`) | 67.85 ± 2.03 | 29.53 ± 3.25 | 71.66 ± 2.61 | 77.43 ± 1.70 |
| Gloss-Free (`0.0 / 0.0`) | 79.24 ± 0.96 | 21.11 ± 1.42 | 81.41 ± 0.91 | 84.92 ± 1.17 |
| CTC + Supervised (`0.3 / 0.3`) | 78.92 ± 2.44 | 20.04 ± 2.69 | 80.64 ± 2.28 | 84.86 ± 1.86 |
| **Supervised Frame CE (`0.0 / 0.3`)** | **82.60 ± 0.31** | **17.11 ± 0.65** | **83.95 ± 0.27** | **87.49 ± 0.44** |
| **(b) Input Features & Fusion** | | | | |
| w/o Face — Pose & Hands only | 76.69 ± 1.19 | 23.60 ± 1.30 | 78.59 ± 0.86 | 82.52 ± 0.80 |
| w/o Velocity | 81.42 ± 0.47 | 18.50 ± 0.55 | 82.95 ± 0.45 | 86.71 ± 0.35 |
| Multi-view Front + Side | 81.32 ± 1.73 | 18.08 ± 1.57 | 82.95 ± 1.01 | 86.80 ± 1.04 |
| **Baseline: Front + Face + Vel.** | **82.60 ± 0.31** | **17.11 ± 0.65** | **83.95 ± 0.27** | **87.49 ± 0.44** |

### Table 8 — LOSO Cross-Validation (TCN Supervised, 300 Sentences)

| Signer | BLEU-4 ↑ | WER (%) ↓ | chrF ↑ | ROUGE-L ↑ |
| :--- | :---: | :---: | :---: | :---: |
| S01 | 80.34 ± 2.93 | 19.07 ± 2.28 | 81.76 ± 2.63 | 85.82 ± 1.80 |
| S02 | 80.01 ± 1.70 | 19.40 ± 1.48 | 81.74 ± 1.74 | 85.70 ± 1.24 |
| S03 | 81.12 ± 1.53 | 19.10 ± 1.36 | 82.43 ± 1.66 | 86.38 ± 0.95 |
| S04 | 79.14 ± 2.66 | 20.49 ± 2.17 | 81.15 ± 2.19 | 85.47 ± 1.29 |
| S05 | 80.92 ± 2.62 | 18.54 ± 1.93 | 82.51 ± 2.12 | 86.34 ± 1.60 |
| S06 | 82.83 ± 1.30 | 17.02 ± 1.03 | 84.13 ± 1.27 | 87.66 ± 0.83 |
| **Avg.** | **80.73 ± 1.96** | **18.94 ± 1.71** | **82.29 ± 1.94** | **86.23 ± 1.29** |

### Table 10 — Downstream Validation under Synthetic Data Scaling (S06 Test Split, 5 Runs)

| Training Strategy | BLEU-4 ↑ | WER (%) ↓ | chrF ↑ | ROUGE-L ↑ |
| :--- | :---: | :---: | :---: | :---: |
| Baseline (Original VSL-GH) | 82.60 ± 0.31 | 17.11 ± 0.65 | 83.95 ± 0.27 | 87.49 ± 0.44 |
| **Synthetic 5× Scaling + Fine-tuning** | **84.75 ± 1.46** | **15.84 ± 1.61** | **85.12 ± 1.35** | **88.75 ± 1.12** |

---

## Ethics & Acknowledgements

- **Ethical Approval & Informed Consent:** All six deaf signers (5 male, 1 female; born 1992–2011; HCMC-raised; identities anonymised as `S01–S06`) provided written informed consent for video recording and open-access research release, with protocol review and support from the **Centre for Research and Education of the Deaf and Hard of Hearing (CED)**.
- **Acknowledgements:** We thank the **Centre for Research and Education of the Deaf and Hard of Hearing (CED)** for expert sign language support throughout data collection and annotation, the physicians at **Tan Hung General Hospital** for clinical validation of the 300 sentences, and the six deaf participants who contributed their time.

---

## Citation

```bibtex
@article{nguyen2026vslgh,
  title   = {VSL-GH: A Sentence-Level Vietnamese Sign Language Dataset for General Health Examination},
  author  = {Nguyen, The Anh},
  year    = {2026},
  url     = {https://github.com/nguyentheanh822/VSL-GH}
}
```

---

## License

This codebase is released under the [MIT License](LICENSE).
