# ============================================================
# Evaluation Script: Flux Reconstruction vs Ground Truth
# ============================================================
# Requirements: pip install zarr pillow numpy scikit-image lpips torch torchvision

import os
import re
import zarr
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import lpips
import torch

from sptflux import config

# ============================================================
# CONFIGURE THESE
# ============================================================
DATA_ROOT    = config.DATA_DIR / "phantom-synth/separated_videos"
RESULTS_ROOT = config.OUTPUT_DIR / "results/phantom-synth"
OUTPUT_CSV   = config.OUTPUT_DIR / "results/phantom-synth" / "metrics.csv"
# RESULTS_ROOT = config.OUTPUT_DIR / "results/uwb1d/phantom-synth"
# OUTPUT_CSV   = config.OUTPUT_DIR / "results/uwb1d/phantom-synth" / "metrics.csv"

method_name = "uwb3d"
# ============================================================

def load_gt_frames_grayscale(dataset_dir: Path) -> np.ndarray:
    """Load all PNG frames from dataset dir, return as (N, H, W) float32 array in [0,1]."""
    png_files = sorted(dataset_dir.glob("*.png"))
    frames = []
    for p in png_files:
        img = np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        # Un-gamma correct
        img = img ** 2.2
        # Weighted grayscale: R=0.289, G=0.587, B=0.114
        gray = 0.289 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        frames.append(gray)
    return np.stack(frames, axis=0)  # (N, H, W)


def normalize_flux(flux: np.ndarray) -> np.ndarray:
    """Normalize flux array to [0, 1]."""
    mn, mx = 0, flux.max()
    if mx == mn:
        return np.zeros_like(flux, dtype=np.float32)
    return ((flux - mn) / (mx - mn)).astype(np.float32)


def parse_thin_level(run_name: str) -> float:
    """Extract thinning level from run name, e.g. 'keep0.250' -> 0.25."""
    match = re.search(r"keep(\d+\.\d+)", run_name)
    if match is None:
        raise ValueError(f"Could not parse thin level from run name: {run_name!r}")
    return float(match.group(1))


def compute_metrics(pred_frames: np.ndarray, gt_frames: np.ndarray, lpips_fn) -> dict:
    """
    Compute mean PSNR, SSIM, and LPIPS across N frames.
    pred_frames, gt_frames: (N, H, W) float32 in [0, 1]
    """
    n = pred_frames.shape[0]
    psnr_vals, ssim_vals, lpips_vals = [], [], []

    for i in range(n):
        p = pred_frames[i]
        g = gt_frames[i]

        psnr_vals.append(psnr(g, p, data_range=1.0))
        ssim_vals.append(ssim(g, p, data_range=1.0))

        # LPIPS expects (1, 3, H, W) tensors in [-1, 1]
        p_t = torch.tensor(p).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1) * 2 - 1
        g_t = torch.tensor(g).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1) * 2 - 1
        with torch.no_grad():
            lpips_vals.append(lpips_fn(p_t, g_t).item())

    return {
        "psnr":  float(np.mean(psnr_vals)),
        "ssim":  float(np.mean(ssim_vals)),
        "lpips": float(np.mean(lpips_vals)),
    }


# ---- Main ----

lpips_fn = lpips.LPIPS(net="alex")
lpips_fn.eval()

records = []

results_root = Path(RESULTS_ROOT)
data_root    = Path(DATA_ROOT)

for dataset_dir in sorted(results_root.iterdir()):
    if not dataset_dir.is_dir():
        continue
    dataset_name = dataset_dir.name
    gt_dir = data_root / dataset_name

    if not gt_dir.exists():
        print(f"[WARN] No ground truth found for dataset '{dataset_name}', skipping.")
        continue

    print(f"\n=== Dataset: {dataset_name} ===")
    gt_frames = load_gt_frames_grayscale(gt_dir)  # (N_gt, H, W)
    print(f"  GT frames: {gt_frames.shape[0]}")

    for run_dir in sorted(dataset_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_name = run_dir.name

        try:
            thin_level = parse_thin_level(run_name)
        except ValueError as e:
            print(f"  [WARN] {e}, skipping.")
            continue
        
        gammamp4_path = run_dir / "gamma.mp4"
        if gammamp4_path.exists():
            # copy to a new name to easily transfer to dropbox
            target_mp4_path = run_dir / f"{method_name}-{dataset_name}-gamma-thin{thin_level:.3f}.mp4"
            print(f"  Found {gammamp4_path}, copying to new name {target_mp4_path}.")
            target_mp4_path.write_bytes(gammamp4_path.read_bytes())

        zarr_path = run_dir / "reconinfo.zarr"
        if not zarr_path.exists():
            print(f"  [WARN] No reconinfo.zarr in {run_dir}, skipping.")
            continue

        try:
            store = zarr.open(str(zarr_path), mode="r")
            flux  = np.array(store["flux"])  # expected shape: (N_flux, H, W) or similar
        except Exception as e:
            print(f"  [WARN] Could not load flux from {zarr_path}: {e}, skipping.")
            continue

        # Flux shape handling: assume first axis is frames
        if flux.ndim == 2:
            flux = flux[np.newaxis]  # treat as single frame
        n_flux = flux.shape[0]

        # Align frame counts (flux may be shorter than gt)
        n_frames = min(n_flux, gt_frames.shape[0])
        flux_trimmed = flux[:n_frames]
        gt_trimmed   = gt_frames[:n_frames]

        print(f"  Run: {run_name} \n| thin={thin_level:.3f} \n| frames used: {n_frames}/{gt_frames.shape[0]}")

        flux_norm = normalize_flux(flux_trimmed)

        # If flux has spatial dims mismatched (e.g. extra channel dim), squeeze
        if flux_norm.ndim == 4 and flux_norm.shape[1] == 1:
            flux_norm = flux_norm[:, 0, :, :]

        metrics = compute_metrics(flux_norm, gt_trimmed, lpips_fn)
        print(f"    PSNR={metrics['psnr']:.4f}  SSIM={metrics['ssim']:.4f}  LPIPS={metrics['lpips']:.4f}")

        records.append({
            "dataset":     dataset_name,
            "thin_level":  thin_level,
            "psnr":        metrics["psnr"],
            "ssim":        metrics["ssim"],
            "lpips":       metrics["lpips"],
        })

df = pd.DataFrame(records, columns=["dataset", "thin_level", "psnr", "ssim", "lpips"])
df = df.sort_values(["dataset", "thin_level"]).reset_index(drop=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved metrics to: {OUTPUT_CSV}")
df
