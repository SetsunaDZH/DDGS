"""
Plot training loss curves for multiple runs with different batch sizes by reading
their TensorBoard event files.

Examples:

1) Compare total loss in fine stage across three runs:
   python tools/plot_loss_vs_batchsize.py \
       --run E:/4DGaussians/output/exp_bs1:bs=1 \
       --run E:/4DGaussians/output/exp_bs2:bs=2 \
       --run E:/4DGaussians/output/exp_bs4:bs=4 \
       --stage fine --metric total_loss --out E:/4DGaussians/output/loss_compare.png

2) Plot L1 loss in coarse stage and show interactively:
   python tools/plot_loss_vs_batchsize.py \
       --run E:/4DGaussians/output/exp_bs1:bs=1 \
       --stage coarse --metric l1_loss --show

Dependencies: pip install matplotlib tensorboard
"""

import os
import argparse
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing import event_accumulator


def parse_run_arg(run_arg: str) -> Tuple[str, str]:
    """Parse a --run argument of the form "/path/to/logdir:label"."""
    if ":" in run_arg:
        path, label = run_arg.split(":", 1)
    else:
        path, label = run_arg, os.path.basename(run_arg.rstrip("/\\"))
    return path, label


def load_scalar(logdir: str, tag: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load (steps, values) arrays for a scalar tag from a TensorBoard logdir.
    Returns empty arrays if tag not found.
    """
    if not os.path.isdir(logdir):
        return np.array([]), np.array([])
    ea = event_accumulator.EventAccumulator(logdir, size_guidance={
        event_accumulator.SCALARS: 0,
    })
    try:
        ea.Reload()
    except Exception:
        return np.array([]), np.array([])

    if tag not in ea.Tags().get("scalars", []):
        return np.array([]), np.array([])

    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events], dtype=np.int64)
    vals = np.array([e.value for e in events], dtype=np.float64)
    return steps, vals


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size == 0:
        return x
    window = min(window, x.size)
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)


def main():
    parser = argparse.ArgumentParser(description="Plot loss curves across batch sizes from TensorBoard logs")
    parser.add_argument("--run", dest="runs", action="append", required=True,
                        help="Run as 'logdir:label'. Provide multiple --run to compare.")
    parser.add_argument("--stage", choices=["coarse", "fine"], default="fine",
                        help="Which stage's tag prefix to read")
    parser.add_argument("--metric", choices=["l1_loss", "total_loss"], default="total_loss",
                        help="Which loss metric to plot")
    parser.add_argument("--smooth", type=int, default=1, help="Moving average window size")
    parser.add_argument("--max-steps", type=int, default=0, help="Trim to at most N steps (0 = no trim)")
    parser.add_argument("--out", type=str, default="", help="Path to save the plot (PNG). If empty, won't save.")
    parser.add_argument("--show", action="store_true", default=False, help="Show the plot interactively")
    args = parser.parse_args()

    # Compose TensorBoard scalar tag aligning with train.py
    if args.metric == "l1_loss":
        tag = f"{args.stage}/train_loss_patches/l1_loss"
    else:
        tag = f"{args.stage}/train_loss_patchestotal_loss"

    series: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for run in args.runs:
        logdir, label = parse_run_arg(run)
        steps, vals = load_scalar(logdir, tag)
        if steps.size == 0:
            print(f"Warning: no data for tag '{tag}' in '{logdir}'")
            continue

        if args.max_steps > 0:
            mask = steps <= args.max_steps
            steps, vals = steps[mask], vals[mask]

        if args.smooth > 1:
            if steps.size >= args.smooth:
                steps = steps[args.smooth - 1 :]
                vals = moving_average(vals, args.smooth)
        series.append((label, steps, vals))

    if not series:
        print("No valid data to plot.")
        return

    plt.figure(figsize=(9, 5))
    for label, steps, vals in series:
        plt.plot(steps, vals, label=label, linewidth=1.8)

    plt.xlabel("Iteration")
    plt.ylabel("Loss" if args.metric == "total_loss" else "L1 Loss")
    plt.title(f"{args.stage} {args.metric} vs iterations (different batch sizes)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(args.out, dpi=150)
        print(f"Saved plot to: {args.out}")

    if args.show or not args.out:
        plt.show()


if __name__ == "__main__":
    main()


