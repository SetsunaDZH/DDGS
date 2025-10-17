# DDGS

DDGS implements dynamic 3D Gaussian splatting with a PyTorch training pipeline. This guide focuses on preparing data and running training jobs.

## Environment setup

1. **Create a Python environment.** Python 3.8+ with CUDA-enabled PyTorch is recommended.
2. **Install PyTorch** following the [official instructions](https://pytorch.org/get-started/locally/).
3. **Install project dependencies:**
   ```bash
   pip install numpy tqdm lpips matplotlib tensorboard mmcv
   ```
   These packages cover the libraries imported by the training and preprocessing scripts. CUDA-capable GPUs with recent NVIDIA drivers are required for practical training runtimes.
4. **Optional tools:**
   - [COLMAP](https://colmap.github.io/) (for structure-from-motion reconstruction)
   - [ImageMagick](https://imagemagick.org/) (only when using image down-sampling during conversion)

## Preparing a scene

1. **Gather frames.** Place the RGB frames of a scene in `<scene_root>/input`.
2. **Run the COLMAP conversion helper:**
   ```bash
   python convert.py --source_path <scene_root> \
       [--camera OPENCV] [--no_gpu] [--skip_matching] [--resize]
   ```
   The script invokes COLMAP to extract/match features, reconstruct, and undistort images, then optionally creates downsampled image pyramids via ImageMagick.
3. **Check the output.** After conversion the scene folder contains subdirectories such as `images/` with undistorted frames and `sparse/0/` with reconstructed camera poses that the training code expects.

## Launching training

The main entry point is `train.py`, which groups command-line options into model, optimization, and pipeline parameters. Key arguments include:

- `--source_path`: absolute path to the processed scene folder.
- `--model_path`: destination directory for checkpoints (optional). If omitted, the trainer creates `./output/<expname>` automatically.
- `--expname`: short name used when auto-creating the output directory.
- `--configs`: path to a Python config that overrides defaults (e.g., files in `arguments/dynerf/`). Config files update optimization or model hyper-parameters via `mmcv` before training begins.

A typical training command looks like:

```bash
python train.py \
    --source_path /data/datasets/dynerf/sear_steak \
    --expname sear_steak_ddgs \
    --configs arguments/dynerf/default.py
```

### What happens during training

1. **Initialization:** The script logs configuration values, initializes random seeds, and starts the renderer GUI server.
2. **Output setup:** Training artifacts, TensorBoard logs, and configuration snapshots are stored under `model_path` (`./output/<expname>` when not specified).
3. **Two-stage optimization:** Each run performs a coarse phase followed by a fine phase using the iteration counts defined in `OptimizationParams` (defaults: 3k coarse, 30k fine, adjustable via CLI or config).
4. **Evaluation & checkpoints:** The trainer renders validation views at iterations listed in `--test_iterations` and saves models at `--save_iterations` plus the final iteration. Optional extra checkpoints can be requested via `--checkpoint_iterations`.
5. **Logging:** If TensorBoard is available, metrics and render previews are written automatically for monitoring.

### Useful flags

- `--test_iterations 3000 7000 14000` and `--save_iterations ...` control validation and snapshot cadence.
- `--detect_anomaly` enables PyTorch anomaly detection for debugging unstable runs.
- `--ip`/`--port` configure the optional live viewer connection.

## After training

The output directory contains:

- Gaussian checkpoint files saved at configured iterations.
- `cfg_args` recording the exact arguments for reproducibility.
- TensorBoard event files (if installed) for loss/PSNR curves and rendered previews.

You can resume training by passing `--start_checkpoint` with the path to a saved checkpoint. Use the provided scripts in `scripts/` as references for dataset-specific workflows once the basic process above is working.
