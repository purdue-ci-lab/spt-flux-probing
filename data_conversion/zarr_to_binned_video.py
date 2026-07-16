#!/usr/bin/env python3
"""
Convert a Zarr v3 dataset (T,H,W) of binary frames into a replay video by
summing every M frames into one "intensity" frame.

- Input: Zarr v3 group/array on disk (directory store), dataset default name "data"
- Output: MP4 (or AVI) video at specified FPS (default 30)
- Aggregation: for each block of M frames:
      I_k(y,x) = sum_{t=k*M:(k+1)*M-1} X_t(y,x)
  producing T_out = floor(T / M) frames

Usage:
  python zarr_to_binned_video.py \
      --zarr /path/to/output.zarr \
      --out  /path/to/replay.mp4 \
      --M 100 \
      --fps 30

Optional:
  --dataset data
  --start 0
  --end 50000          # end frame index in original T (exclusive)
  --normalize auto|clip|none
  --codec mp4v         # or avc1, H264 if your OpenCV build supports it
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Optional

import numpy as np


def to_uint8_frame(
    summed: np.ndarray,
    M: int,
    normalize: Literal["auto", "clip", "none"] = "auto",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """
    Convert summed counts to uint8 for video.
    - clip: clamp to [0,M] then scale to [0,255]
    - none: assumes already in [0,255] range, just cast (dangerous)
    - auto: uses robust percentile scaling per-frame (good when dynamic range varies)
    """
    x = summed.astype(np.float32, copy=False)

    if normalize == "clip":
        x = np.clip(x, 0.0, float(M))
        x = (x / float(M)) * 255.0

    elif normalize == "auto":
        # robust scaling using percentiles (avoids single hot pixels ruining contrast)
        lo = np.percentile(x, 1.0) if vmin is None else float(vmin)
        hi = np.percentile(x, 99.5) if vmax is None else float(vmax)
        if hi <= lo + 1e-6:
            hi = lo + 1.0
        x = (x - lo) / (hi - lo)
        x = np.clip(x, 0.0, 1.0) * 255.0

    elif normalize == "none":
        # no scaling; just cast
        pass
    else:
        raise ValueError(f"Unknown normalize mode: {normalize}")

    return x.astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", required=True, type=str, help="Path to Zarr v3 group (e.g., output.zarr)")
    ap.add_argument("--out", required=True, type=str, help="Output video path (.mp4 or .avi)")
    ap.add_argument("--dataset", default="data", type=str, help="Dataset name inside the Zarr group (default: data)")
    ap.add_argument("--M", default=100, type=int, help="Bin size (#frames to sum per output frame). Default: 100")
    ap.add_argument("--fps", default=30, type=float, help="Output video FPS. Default: 30")
    ap.add_argument("--start", default=0, type=int, help="Start frame (in original T). Default: 0")
    ap.add_argument("--end", default=None, type=int, help="End frame (exclusive, in original T). Default: T")
    ap.add_argument(
        "--normalize",
        default="clip",
        choices=["auto", "clip", "none"],
        help="How to map summed counts to uint8. Default: clip (0..M -> 0..255).",
    )
    ap.add_argument("--vmin", default=None, type=float, help="Optional vmin for auto normalization")
    ap.add_argument("--vmax", default=None, type=float, help="Optional vmax for auto normalization")
    ap.add_argument("--codec", default="mp4v", type=str, help="FourCC codec (e.g., mp4v, avc1). Default: mp4v")
    ap.add_argument("--progress", action="store_true", help="Print progress.")
    args = ap.parse_args()

    # Zarr v3 load
    import zarr  # type: ignore

    zarr_path = Path(args.zarr).expanduser()
    out_path = Path(args.out).expanduser()
    M = int(args.M)
    if M <= 0:
        raise ValueError("--M must be > 0")

    root = zarr.open_group(zarr_path.as_posix(), mode="r")
    z = root[args.dataset]  # expects shape (T,H,W)

    if z.ndim != 3:
        raise ValueError(f"Expected dataset with 3 dims (T,H,W). Got shape {z.shape}")

    T, H, W = map(int, z.shape)

    start = int(args.start)
    end = int(args.end) if args.end is not None else T
    if not (0 <= start < T):
        raise ValueError(f"--start must be in [0,{T-1}]")
    if not (start < end <= T):
        raise ValueError(f"--end must be in ({start},{T}]")

    # Truncate to a multiple of M so the output frame count is exactly floor((end-start)/M)
    usable = end - start
    T_out = usable // M
    end_eff = start + T_out * M
    if T_out <= 0:
        raise ValueError(f"Range too small for M={M}. (end-start)={usable}")

    # Video writer
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise SystemExit("OpenCV not installed. Install with: pip install opencv-python") from e

    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(out_path.as_posix(), fourcc, float(args.fps), (W, H), isColor=False)
    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open video writer for {out_path}. "
            f"Try a different --codec (e.g., avc1) or output .avi with codec XVID."
        )

    if args.progress:
        print(f"Input: {zarr_path} dataset='{args.dataset}' shape={z.shape} dtype={z.dtype}")
        print(f"Output: {out_path} fps={args.fps} codec={args.codec}")
        print(f"start={start}, end={end} -> using end_eff={end_eff} (multiple of M)")
        print(f"M={M} => output frames T_out={T_out} (expected size ~ T/M)")

    # Process in blocks of M frames
    # Note: z[start_i:end_i] reads a contiguous time chunk efficiently when chunks align with time.
    for k in range(T_out):
        t0 = start + k * M
        t1 = t0 + M

        # Read chunk: (M,H,W)
        chunk = z[t0:t1, :, :]  # zarr returns array-like; numpy ops will materialize
        # Sum over time -> (H,W)
        summed = np.asarray(chunk).sum(axis=0)

        frame_u8 = to_uint8_frame(
            summed,
            M=M,
            normalize=args.normalize,
            vmin=args.vmin,
            vmax=args.vmax,
        )

        # OpenCV expects 2D uint8 for grayscale when isColor=False; some builds want (H,W,1)
        ok = writer.write(frame_u8)
        # cv2.VideoWriter.write returns None in many builds; so no check

        if args.progress and (k % 50 == 0 or k == T_out - 1):
            print(f"wrote {k+1}/{T_out} (t={t0}:{t1})")

    writer.release()
    if args.progress:
        print("Done.")


if __name__ == "__main__":
    main()