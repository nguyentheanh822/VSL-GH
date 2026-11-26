"""Table 10 (Section 4.6): Downstream Validation under Synthetic 5x Data Scaling.

Constructs a 5x synthetic training set strictly from the training signers (S01-S04)
using offline spatial augmentations (Gaussian noise, 2D skeleton rotation, body scaling)
and temporal augmentations (speed perturbation with proportional gloss-boundary scaling),
leaving validation (S05) and test (S06) splits completely untouched.

Two-stage training strategy:
  Stage 1: Pre-train TCN on the synthetic 5x training set.
  Stage 2: Fine-tune on the original VSL-GH training set (S01-S04).
  Eval   : Evaluate on the unseen S06 test split with beam search (beam_size=4) across 5 seeds.

Examples:
    # Full pipeline (generate 5x synthetic train split + 5-seed pretrain/finetune/eval)
    python scripts/run_synthetic_scaling.py --gpu 0 --seeds 42 43 44 45 46

    # Only generate the offline 5x synthetic training dataset
    python scripts/run_synthetic_scaling.py --generate_only

    # Run a single seed after the synthetic set is already generated
    python scripts/run_synthetic_scaling.py --skip_generate --gpu 0 --seeds 42
"""
import argparse
import copy
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def augment_keypoint_sequence(
    kp: np.ndarray,
    rng: np.random.Generator,
    noise_std: float = 0.02,
    rotation_deg: float = 20.0,
    scale_range: tuple[float, float] = (0.85, 1.15),
    speed_range: tuple[float, float] = (0.6, 1.4),
) -> tuple[np.ndarray, float]:
    """Apply offline spatial + temporal augmentations to a raw (T, 411) keypoint array."""
    arr = kp.astype(np.float32, copy=True)

    # 1. Temporal speed perturbation
    speed = float(rng.uniform(speed_range[0], speed_range[1]))
    T = len(arr)
    new_T = max(2, int(round(T / speed)))
    indices = np.linspace(0, T - 1, new_T).astype(int)
    arr = arr[indices]

    # 2. 2D skeleton rotation in (x, y) around frame mean (non-zero landmarks)
    deg = float(rng.uniform(-rotation_deg, rotation_deg))
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    xs = arr[:, 0::3]
    ys = arr[:, 1::3]
    cx = np.mean(xs, axis=1, keepdims=True)
    cy = np.mean(ys, axis=1, keepdims=True)
    dx, dy = xs - cx, ys - cy
    arr[:, 0::3] = dx * c - dy * s + cx
    arr[:, 1::3] = dx * s + dy * c + cy

    # 3. Body scaling on (x, y)
    scale = float(rng.uniform(scale_range[0], scale_range[1]))
    arr[:, 0::3] = (arr[:, 0::3] - cx) * scale + cx
    arr[:, 1::3] = (arr[:, 1::3] - cy) * scale + cy

    # 4. Gaussian jitter noise
    if noise_std > 0:
        arr += rng.normal(0.0, noise_std, size=arr.shape).astype(np.float32)

    return arr, speed


def scale_boundaries(boundaries: list[dict], speed: float, max_frame: int) -> list[dict]:
    """Scale frame-level gloss boundaries proportionally to temporal speed perturbation."""
    scaled = []
    for b in boundaries:
        nb = dict(b)
        if "start" in nb and "end" in nb:
            st = max(0, min(max_frame - 1, int(round(nb["start"] / speed))))
            en = max(st + 1, min(max_frame, int(round(nb["end"] / speed))))
            nb["start"] = st
            nb["end"] = en
        scaled.append(nb)
    return scaled


def generate_synthetic_5x(
    dataset_json: Path,
    keypoints_dir: Path,
    out_json: Path,
    scale_factor: int = 5,
    seed: int = 42,
):
    """Generate offline (scale_factor)x training set strictly from train signers (S01-S04)."""
    entries = json.loads(dataset_json.read_text(encoding="utf-8"))
    synth_sub = "synthetic_5x"
    synth_dir = keypoints_dir / synth_sub
    synth_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    out_entries = []
    n_train_orig = 0
    n_synth = 0

    for item in entries:
        out_entries.append(item)
        if item.get("split") != "train":
            # Validation (S05) and Test (S06) remain completely untouched
            continue

        n_train_orig += 1
        src_id = item["id"].strip()
        src_kp_path = keypoints_dir / f"{src_id}.npy"
        if not src_kp_path.exists():
            raise FileNotFoundError(f"Missing keypoint file: {src_kp_path}")
        kp = np.load(src_kp_path)

        for k in range(1, scale_factor):
            aug_id = f"{synth_sub}/{src_id}_aug{k}"
            aug_kp_path = keypoints_dir / f"{aug_id}.npy"
            aug_kp, speed = augment_keypoint_sequence(kp, rng)
            np.save(aug_kp_path, aug_kp)

            aug_item = copy.deepcopy(item)
            aug_item["id"] = aug_id
            if aug_item.get("gloss_frame_boundaries"):
                aug_item["gloss_frame_boundaries"] = scale_boundaries(
                    aug_item["gloss_frame_boundaries"], speed, len(aug_kp)
                )
            out_entries.append(aug_item)
            n_synth += 1

    out_json.write_text(json.dumps(out_entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[Synthetic {scale_factor}x] Original train={n_train_orig}, Generated={n_synth}, "
        f"Total train={n_train_orig + n_synth}, Total entries (incl. untouched val/test)={len(out_entries)}",
        flush=True,
    )
    print(f"[Synthetic {scale_factor}x] Saved manifest to {out_json}", flush=True)


def summarize(out_root: Path, seeds: list[int]):
    metrics = {k: [] for k in ("bleu4", "wer", "chrf", "rouge_l")}
    for s in seeds:
        ev = out_root / f"seed{s}" / "eval_beam4.json"
        if ev.exists():
            d = json.loads(ev.read_text(encoding="utf-8"))
            for k in metrics:
                if k in d:
                    metrics[k].append(float(d[k]))
    n = len(metrics["bleu4"])
    if n == 0:
        return
    summary = {
        "n": n,
        **{f"{k}_mean": float(np.mean(v)) for k, v in metrics.items()},
        **{f"{k}_std": float(np.std(v)) for k, v in metrics.items()},
    }
    text = (
        f"Table 10 — Synthetic 5x Scaling + Fine-tuning (n={n}, beam_size=4, S06 test)\n"
        f"  BLEU-4  : {summary['bleu4_mean']:.2f} ± {summary['bleu4_std']:.2f}\n"
        f"  WER (%) : {summary['wer_mean']:.2f} ± {summary['wer_std']:.2f}\n"
        f"  chrF    : {summary['chrf_mean']:.2f} ± {summary['chrf_std']:.2f}\n"
        f"  ROUGE-L : {summary['rouge_l_mean']:.2f} ± {summary['rouge_l_std']:.2f}\n"
    )
    (out_root / "summary.txt").write_text(text, encoding="utf-8")
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(text, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Table 10: Synthetic 5x Scaling + Fine-tuning on VSL-GH.")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--scale_factor", type=int, default=5, help="Expansion multiplier for train split (default: 5).")
    parser.add_argument("--beam_size", type=int, default=4, help="Beam size for test decoding (paper uses 4).")
    parser.add_argument("--out_dir", type=str, default="results/table10_synthetic_scaling")
    parser.add_argument("--generate_only", action="store_true", help="Only generate synthetic 5x dataset and exit.")
    parser.add_argument("--skip_generate", action="store_true", help="Skip generation if data/dataset_synthetic_5x.json exists.")
    args = parser.parse_args()

    kp_dir = ROOT / "data/keypoints"
    if not kp_dir.exists() or not any(kp_dir.glob("*.npy")):
        print(f"[ERROR] No keypoint .npy files found in {kp_dir}. See data/README.md.", file=sys.stderr)
        sys.exit(1)

    synth_json = ROOT / "data/dataset_synthetic_5x.json"
    if not args.skip_generate or not synth_json.exists():
        generate_synthetic_5x(
            dataset_json=ROOT / "data/dataset.json",
            keypoints_dir=kp_dir,
            out_json=synth_json,
            scale_factor=args.scale_factor,
            seed=args.seeds[0],
        )

    if args.generate_only:
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["OMP_NUM_THREADS"] = "4"

    out_root = ROOT / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    cfg_pre = ROOT / "configs/table10_synthetic_scaling/tcn_pretrain_5x.yaml"
    cfg_ft = ROOT / "configs/table10_synthetic_scaling/tcn_finetune.yaml"

    for seed in args.seeds:
        seed_dir = out_root / f"seed{seed}"
        pre_dir = seed_dir / "stage1_pretrain_5x"
        ft_dir = seed_dir / "stage2_finetune"
        pre_dir.mkdir(parents=True, exist_ok=True)
        ft_dir.mkdir(parents=True, exist_ok=True)

        pre_ckpt = pre_dir / "checkpoint_best.pt"
        pre_flag = pre_dir / "train_done.flag"
        ft_ckpt = ft_dir / "checkpoint_best.pt"
        ft_flag = ft_dir / "train_done.flag"
        eval_out = seed_dir / "eval_beam4.json"

        # Stage 1: Pre-train on Synthetic 5x
        if not (pre_ckpt.exists() and pre_flag.exists()):
            cmd_pre = [
                sys.executable, "-u", str(ROOT / "source/train.py"),
                "--config", str(cfg_pre),
                "--seed", str(seed),
                "--base_dir", str(ROOT),
                "--save_dir", str(pre_dir.relative_to(ROOT)),
            ]
            print(f"[seed {seed}] Stage 1: Pre-training TCN on Synthetic {args.scale_factor}x...", flush=True)
            with open(pre_dir / "train.log", "w", encoding="utf-8") as lf:
                rc = subprocess.call(cmd_pre, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
            if rc != 0 or not pre_ckpt.exists():
                print(f"[seed {seed}] Stage 1 failed (rc={rc})", flush=True)
                continue
            pre_flag.write_text("ok", encoding="utf-8")

        # Stage 2: Fine-tune on Original VSL-GH
        if not (ft_ckpt.exists() and ft_flag.exists()):
            cmd_ft = [
                sys.executable, "-u", str(ROOT / "source/train.py"),
                "--config", str(cfg_ft),
                "--init_from", str(pre_ckpt),
                "--seed", str(seed),
                "--base_dir", str(ROOT),
                "--save_dir", str(ft_dir.relative_to(ROOT)),
            ]
            print(f"[seed {seed}] Stage 2: Fine-tuning TCN on original VSL-GH...", flush=True)
            with open(ft_dir / "train.log", "w", encoding="utf-8") as lf:
                rc = subprocess.call(cmd_ft, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
            if rc != 0 or not ft_ckpt.exists():
                print(f"[seed {seed}] Stage 2 failed (rc={rc})", flush=True)
                continue
            ft_flag.write_text("ok", encoding="utf-8")

        # Evaluate on unseen S06 test split with beam search (beam_size=4)
        cmd_ev = [
            sys.executable, "-u", str(ROOT / "source/evaluate.py"),
            "--config", str(cfg_ft),
            "--checkpoint", str(ft_ckpt),
            "--split", "test",
            "--beam_size", str(args.beam_size),
            "--base_dir", str(ROOT),
            "--output", str(eval_out),
        ]
        print(f"[seed {seed}] Evaluating on S06 test split (beam_size={args.beam_size})...", flush=True)
        with open(seed_dir / "eval.log", "w", encoding="utf-8") as lf:
            rc = subprocess.call(cmd_ev, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
        if rc != 0:
            print(f"[seed {seed}] Eval failed (rc={rc})", flush=True)

        summarize(out_root, args.seeds)

    summarize(out_root, args.seeds)


if __name__ == "__main__":
    main()
