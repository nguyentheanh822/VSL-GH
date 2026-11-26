# VSL-GH Data Directory

Because the extracted keypoint arrays (8,400 `.npy` files, ~1.2 GB) and raw videos exceed standard Git repository limits, they are hosted externally.

## Expected Directory Layout

Download the VSL-GH dataset archive (see link in the root [README.md](../README.md)) and extract it so that this directory has the following structure:

```text
data/
├── dataset.json                # Included: 4,200 frontal entries with frame-aligned gloss boundaries
├── dataset_loso_s01.json       # Included: LOSO fold 1 (S01 held out for test)
├── dataset_loso_s02.json       # Included: LOSO fold 2 (S02 held out for test)
├── dataset_loso_s03.json       # Included: LOSO fold 3 (S03 held out for test)
├── dataset_loso_s04.json       # Included: LOSO fold 4 (S04 held out for test)
├── dataset_loso_s05.json       # Included: LOSO fold 5 (S05 held out for test)
├── dataset_loso_s06.json       # Included: LOSO fold 6 (S06 held out for test)
├── gloss_vocab.txt             # Included: Full 300-sentence gloss vocabulary (369 glosses + 5 special tokens)
├── trans_vocab.txt             # Included: Vietnamese translation vocabulary
├── splits/                     # Included: Signer-independent split lists (S01-S04 train, S05 val, S06 test)
├── keypoints/                  # Download externally: 8,400 MediaPipe Holistic .npy files (4,200 *_F.npy + 4,200 *_S.npy)
└── annotations/                # Download externally: 4,200 ELAN-exported temporal gloss annotation .txt files
```

Each `.npy` file in `data/keypoints/` has shape `(T, 137, 3)` corresponding to 25 body pose, 70 facial, and 42 hand 3D landmarks extracted with MediaPipe Holistic at 30 FPS.
