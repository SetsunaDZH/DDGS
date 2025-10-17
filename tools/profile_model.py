"""
Profile parameter counts (and optional FLOPs) for the Gaussian model setup.

Usage examples:

1) Count parameters after full scene/model initialization (recommended):
   python tools/profile_model.py --configs path/to/config.py --expname test_profile

2) Optionally attempt FLOPs on deformation network (if its forward signature is known):
   python tools/profile_model.py --configs path/to/config.py --flops-deformation --flops-batch 1024 --flops-dim 3

Notes:
- This script follows train.py to instantiate dataset, scene, and gaussians so that
  all nn.Parameters (e.g., _xyz, _features_*, _opacity, etc.) are materialized.
- FLOPs are only attempted for the deformation sub-module if it is a torch.nn.Module
  with a simple forward signature. You can adjust dummy input size via flags.
"""

import sys
import os
import argparse
import torch

# Reuse the same argument helpers used by train.py
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from scene import Scene, GaussianModel


def count_tensor_parameters(t: torch.Tensor) -> int:
    if not isinstance(t, torch.Tensor):
        return 0
    try:
        return int(t.numel())
    except Exception:
        return 0


def count_gaussian_parameters(gaussians: GaussianModel) -> dict:
    """Return a breakdown and total of parameter counts within GaussianModel.

    Includes both raw nn.Parameter fields on the GaussianModel instance and
    parameters of any nested torch.nn.Module (e.g., deformation network).
    """
    breakdown = {}

    # Known nn.Parameter attributes commonly used
    candidate_attrs = [
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_scaling",
        "_rotation",
        "_opacity",
    ]

    for name in candidate_attrs:
        p = getattr(gaussians, name, None)
        breakdown[name] = count_tensor_parameters(p)

    # Deformation module parameters (if module present)
    deform_params = 0
    deform = getattr(gaussians, "_deformation", None)
    if isinstance(deform, torch.nn.Module):
        deform_params = sum(int(p.numel()) for p in deform.parameters() if p is not None)
    breakdown["_deformation_module"] = deform_params

    total = sum(breakdown.values())
    breakdown["total"] = total
    return breakdown


def try_compute_flops_for_deformation(gaussians: GaussianModel, batch: int, dim: int) -> tuple:
    """Attempt to compute FLOPs for the deformation sub-network using thop.

    Returns (macs, params) if successful, otherwise (None, None).
    """
    deform = getattr(gaussians, "_deformation", None)
    if not isinstance(deform, torch.nn.Module):
        return (None, None)

    try:
        from thop import profile  # type: ignore
    except Exception:
        return (None, None)

    # Heuristic dummy input. Adjust if your network expects a different shape.
    # Commonly deformation networks take 3D positions (x,y,z) + optional time.
    dummy = torch.randn(batch, dim, device="cuda" if torch.cuda.is_available() else "cpu")
    device = next(deform.parameters()).device if any(True for _ in deform.parameters()) else dummy.device
    deform = deform.to(device)
    dummy = dummy.to(device)

    try:
        macs, params = profile(deform, inputs=(dummy,), verbose=False)
        return (int(macs), int(params))
    except Exception:
        return (None, None)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Profile Gaussian model parameters and FLOPs")

    # Mirror training args helpers
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)

    parser.add_argument("--configs", type=str, default="", help="mmcv-style config file path (optional)")
    parser.add_argument("--expname", type=str, default="profile_run")
    parser.add_argument("--start-checkpoint", type=str, default=None)

    # FLOPs options for deformation module
    parser.add_argument("--flops-deformation", action="store_true", default=False,
                        help="Attempt FLOPs for _deformation sub-network using thop")
    parser.add_argument("--flops-batch", type=int, default=1024, help="Dummy batch size for FLOPs input")
    parser.add_argument("--flops-dim", type=int, default=3, help="Dummy feature dimension for FLOPs input")

    args = parser.parse_args(argv)

    # Optional config merging (like train.py)
    if args.configs:
        import mmcv  # type: ignore
        from utils.params_utils import merge_hparams
        cfg = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, cfg)

    # Build dataset/scene/model just like in training so parameters are created
    gaussians = GaussianModel(lp.extract(args).sh_degree, hp.extract(args))
    dataset = lp.extract(args)
    dataset.model_path = args.model_path

    # Instantiating Scene triggers model initialization (e.g., create_from_pcd)
    _ = Scene(dataset, gaussians, load_coarse=None)

    # Parameter counting
    breakdown = count_gaussian_parameters(gaussians)

    print("Parameter count (by component):")
    for k, v in breakdown.items():
        if k == "total":
            continue
        print(f"  {k}: {v}")
    print(f"Total parameters: {breakdown['total']}")

    # Optional FLOPs on deformation sub-module
    if args.flops_deformation:
        macs, params = try_compute_flops_for_deformation(gaussians, args.flops_batch, args.flops_dim)
        if macs is None:
            print("FLOPs (deformation): unavailable (install thop or adjust dummy input/signature)")
        else:
            # Report MACs and approximate FLOPs (2*MACs) convention
            print(f"Deformation MACs: {macs}")
            print(f"Deformation approx FLOPs: {macs * 2}")
            print(f"Deformation params (thop): {params}")


if __name__ == "__main__":
    sys.exit(main())


