"""Reproducible multi-seed benchmark runner for VSL-GH (Tables VI, VIIa, VIIb, VIII).

Examples:
    # Run Table VI (Architecture Benchmark) across 5 seeds on GPU 0
    python scripts/run_benchmark.py --suite table6 --gpu 0 --seeds 42 43 44 45 46

    # Run all ablations (Tables VIIa & VIIb)
    python scripts/run_benchmark.py --suite table7 --gpu 0 --seeds 42 43 44 45 46

    # Run Table VIII (6-Fold LOSO Cross-Validation)
    python scripts/run_benchmark.py --suite table8 --gpu 0 --seeds 42 43 44 45 46

    # Run a single experiment (e.g., TCN baseline, 1 seed)
    python scripts/run_benchmark.py --only tcn --gpu 0 --seeds 42
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

EXPERIMENTS = {
    # Table VI: Architecture Benchmark (all share Supervised Frame CE weight=0.3)
    "tcn":         ("table6", "configs/table6_architecture/tcn.yaml", "test"),
    "conformer":   ("table6", "configs/table6_architecture/conformer.yaml", "test"),
    "mamba":       ("table6", "configs/table6_architecture/mamba.yaml", "test"),
    "transformer": ("table6", "configs/table6_architecture/transformer.yaml", "test"),
    "stgcn":       ("table6", "configs/table6_architecture/stgcn.yaml", "test"),
    # Table VIIa: Gloss Supervision Ablation (tcn is Supervised Frame CE baseline)
    "gloss_free":  ("table7a", "configs/table7a_supervision/gloss_free.yaml", "test"),
    "ctc_sup":     ("table7a", "configs/table7a_supervision/ctc_supervised.yaml", "test"),
    "ctc_only":    ("table7a", "configs/table7a_supervision/ctc_only.yaml", "test"),
    # Table VIIb: Input Features & View Ablation (tcn is Full Frontal baseline)
    "no_velocity": ("table7b", "configs/table7b_ablation/no_velocity.yaml", "test"),
    "no_face":     ("table7b", "configs/table7b_ablation/no_face.yaml", "test"),
    "multi_view":  ("table7b", "configs/table7b_ablation/multi_view.yaml", "test"),
    # Table VIII: 6-Fold LOSO Cross-Signer Evaluation
    "loso_s01":    ("table8", "configs/table8_loso/loso_s01.yaml", "test"),
    "loso_s02":    ("table8", "configs/table8_loso/loso_s02.yaml", "test"),
    "loso_s03":    ("table8", "configs/table8_loso/loso_s03.yaml", "test"),
    "loso_s04":    ("table8", "configs/table8_loso/loso_s04.yaml", "test"),
    "loso_s05":    ("table8", "configs/table8_loso/loso_s05.yaml", "test"),
    "loso_s06":    ("table8", "configs/table8_loso/loso_s06.yaml", "test"),
}


def select_experiments(suite: str, only: list[str] | None) -> list[str]:
    if only:
        unknown = [x for x in only if x not in EXPERIMENTS]
        if unknown:
            raise ValueError(f"Unknown experiment(s): {unknown}. Valid: {list(EXPERIMENTS.keys())}")
        return only
    if suite == "all":
        return list(EXPERIMENTS.keys())
    if suite == "table7":
        return ["tcn", "gloss_free", "ctc_sup", "ctc_only", "no_velocity", "no_face", "multi_view"]
    return [k for k, (g, _, _) in EXPERIMENTS.items() if g == suite]


def summarize(out_root: Path, names: list[str], seeds: list[int]):
    lines = [
        f"{'Experiment':<16} | {'n':>2} | {'BLEU-4':>13} | {'WER (%)':>13} | {'chrF':>13} | {'ROUGE-L':>13}",
        "-" * 84,
    ]
    summary = {}
    for name in names:
        metrics = {k: [] for k in ("bleu4", "wer", "chrf", "rouge_l")}
        for s in seeds:
            ev = out_root / f"{name}_seed{s}" / "eval_predictions.json"
            if ev.exists():
                d = json.loads(ev.read_text(encoding="utf-8"))
                for k in metrics:
                    if k in d:
                        metrics[k].append(float(d[k]))
        n = len(metrics["bleu4"])
        if n == 0:
            continue
        summary[name] = {
            "n": n,
            **{f"{k}_mean": float(np.mean(v)) for k, v in metrics.items()},
            **{f"{k}_std": float(np.std(v)) for k, v in metrics.items()},
        }
        lines.append(
            f"{name:<16} | {n:2d} | "
            f"{np.mean(metrics['bleu4']):6.2f}±{np.std(metrics['bleu4']):5.2f} | "
            f"{np.mean(metrics['wer']):6.2f}±{np.std(metrics['wer']):5.2f} | "
            f"{np.mean(metrics['chrf']):6.2f}±{np.std(metrics['chrf']):5.2f} | "
            f"{np.mean(metrics['rouge_l']):6.2f}±{np.std(metrics['rouge_l']):5.2f}"
        )
    text = "\n".join(lines) + "\n"
    (out_root / "summary.txt").write_text(text, encoding="utf-8")
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(text, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run VSL-GH multi-seed benchmarks.")
    parser.add_argument("--suite", default="all", choices=["all", "table6", "table7", "table7a", "table7b", "table8"])
    parser.add_argument("--only", nargs="+", default=None, help="Specific experiment names to run.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--out_dir", type=str, default="results/benchmark", help="Output directory relative to repo root.")
    parser.add_argument("--summarize_only", action="store_true", help="Only aggregate existing results.")
    args = parser.parse_args()

    out_root = ROOT / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    names = select_experiments(args.suite, args.only)

    if args.summarize_only:
        summarize(out_root, names, args.seeds)
        return

    kp_dir = ROOT / "data" / "keypoints"
    if not kp_dir.exists() or not any(kp_dir.glob("*.npy")):
        print(
            f"[ERROR] No keypoint .npy files found in {kp_dir}.\n"
            "Please download the VSL-GH keypoints archive and extract it to data/keypoints/ (see data/README.md).",
            file=sys.stderr,
        )
        sys.exit(1)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["OMP_NUM_THREADS"] = "4"

    jobs = [(s, name) for s in args.seeds for name in names]
    for idx, (seed, name) in enumerate(jobs, 1):
        _, cfg_rel, split = EXPERIMENTS[name]
        run_dir = out_root / f"{name}_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt = run_dir / "checkpoint_best.pt"
        done_flag = run_dir / "train_done.flag"
        eval_out = run_dir / "eval_predictions.json"

        if eval_out.exists() and done_flag.exists():
            print(f"[{idx}/{len(jobs)}] [skip] {name}_seed{seed} (already done)", flush=True)
            continue

        if not (ckpt.exists() and done_flag.exists()):
            cmd_train = [
                sys.executable, "-u", str(ROOT / "source/train.py"),
                "--config", str(ROOT / cfg_rel),
                "--seed", str(seed),
                "--base_dir", str(ROOT),
                "--save_dir", str(run_dir.relative_to(ROOT)),
            ]
            print(f"[{idx}/{len(jobs)}] [train] {name}_seed{seed} (GPU {args.gpu})", flush=True)
            with open(run_dir / "train.log", "w", encoding="utf-8") as lf:
                rc = subprocess.call(cmd_train, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
            if rc != 0 or not ckpt.exists():
                print(f"[{idx}/{len(jobs)}] [ERROR] {name}_seed{seed} failed with rc={rc}", flush=True)
                continue
            done_flag.write_text("ok", encoding="utf-8")

        cmd_eval = [
            sys.executable, "-u", str(ROOT / "source/evaluate.py"),
            "--config", str(ROOT / cfg_rel),
            "--checkpoint", str(ckpt),
            "--split", split,
            "--base_dir", str(ROOT),
            "--output", str(eval_out),
        ]
        print(f"[{idx}/{len(jobs)}] [eval] {name}_seed{seed} (split={split})", flush=True)
        with open(run_dir / "eval.log", "w", encoding="utf-8") as lf:
            rc = subprocess.call(cmd_eval, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
        if rc != 0:
            print(f"[{idx}/{len(jobs)}] [ERROR] eval {name}_seed{seed} failed with rc={rc}", flush=True)

        summarize(out_root, names, args.seeds)

    summarize(out_root, names, args.seeds)


if __name__ == "__main__":
    main()
