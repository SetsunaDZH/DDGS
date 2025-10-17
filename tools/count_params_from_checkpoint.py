"""
Count parameter sizes directly from a saved training checkpoint (.pth) on CPU.

This avoids GPU/CUDA requirements by loading with map_location="cpu".

Usage:
  python tools/count_params_from_checkpoint.py --ckpt E:/4DGaussians/output/exp/chkpnt_fine_14000.pth

The training code saves checkpoints via:
  torch.save((gaussians.capture(), iteration), ...)

We therefore expect the checkpoint to be a tuple (model_params, iteration),
where model_params is typically a dict-like structure of tensors and lists.
"""

import argparse
import os
import sys
from typing import Any, Dict, Tuple

import torch


def flatten_tensors(obj: Any, prefix: str = "") -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    if isinstance(obj, torch.Tensor):
        out[prefix or "tensor"] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            name = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_tensors(v, name))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            name = f"{prefix}[{i}]" if prefix else f"[{i}]"
            out.update(flatten_tensors(v, name))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Count parameters in a checkpoint (CPU)")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .pth file")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.ckpt):
        print(f"Checkpoint not found: {args.ckpt}")
        sys.exit(1)

    data = torch.load(args.ckpt, map_location="cpu")

    # Expected format: (model_params, iteration)
    if isinstance(data, tuple) and len(data) >= 1:
        model_params = data[0]
    else:
        model_params = data

    flat = flatten_tensors(model_params)
    total = 0
    print("Parameter sizes by key:")
    for k in sorted(flat.keys()):
        t = flat[k]
        num = t.numel()
        total += num
        shape_str = "x".join(str(d) for d in t.shape)
        print(f"  {k}: {shape_str} -> {num}")

    print(f"Total parameters (elements): {total}")


if __name__ == "__main__":
    sys.exit(main())


