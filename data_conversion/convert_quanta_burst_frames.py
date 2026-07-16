#!/usr/bin/env python3
"""
Concatenate WxHxT .mat volumes (alphabetical *natural* order) into a single Zarr v3
array stored as TxHxW, with compression + chunking.

- Reads metadata JSON like:
  {
    "no_frames_total": 130000,
    "no_parts": 13,
    "no_frames": 10000
  }

- Assumes each .mat contains ONE main 3D array (WxHxT) unless --key is provided.
- Writes output as Zarr (directory store) so it's compressed and supports partial reads.

Tested for Zarr v3 (e.g. zarr==3.1.x).

Usage:
  python concat_mat_to_zarr_v3.py \
      --folder /path/to/mats \
      --meta /path/to/meta.json \
      --out /path/to/output.zarr

Optional:
  --key VAR_NAME
  --dtype uint16|float32|...
  --chunks 128 512 512          # (T,H,W) chunk shape
  --clevel 5                    # zstd compression level (via Blosc zstd)
  --overwrite                   # overwrite existing output
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def natural_key(s: str):
    # "part12.mat" -> ["part", 12, ".mat"]
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def load_mat_array(mat_path: Path, key: Optional[str] = None) -> np.ndarray:
    """
    Load a 3D array from .mat.
    Supports:
      - MATLAB v7 / v7.2 via scipy.io.loadmat
      - MATLAB v7.3 via h5py (HDF5)

    Returns:
      np.ndarray with shape (W, H, T) as stored in the file.
    """
    # Try scipy first (non-v7.3)
    try:
        import scipy.io  # type: ignore
        d = scipy.io.loadmat(mat_path.as_posix())
        candidates = [k for k in d.keys() if not k.startswith("__")]
        if not candidates:
            raise ValueError(f"No array variables found in {mat_path.name}")

        if key is None:
            if len(candidates) != 1:
                raise ValueError(
                    f"{mat_path.name}: multiple variables found {candidates}. "
                    f"Specify one with --key."
                )
            key_use = candidates[0]
        else:
            if key not in d:
                raise KeyError(f"{mat_path.name}: key '{key}' not found. Available: {candidates}")
            key_use = key

        arr = np.asarray(d[key_use])
        return arr

    except NotImplementedError:
        pass
    except Exception as e:
        scipy_error = e
    else:
        scipy_error = None

    # Try h5py (v7.3)
    try:
        import h5py  # type: ignore
        with h5py.File(mat_path.as_posix(), "r") as f:
            if key is None:
                datasets = []

                def _visit(name, obj):
                    if hasattr(obj, "shape") and hasattr(obj, "dtype") and len(getattr(obj, "shape", ())) == 3:
                        datasets.append(name)

                f.visititems(_visit)

                if not datasets:
                    raise ValueError(f"{mat_path.name}: no 3D datasets found (v7.3).")
                if len(datasets) != 1:
                    raise ValueError(
                        f"{mat_path.name}: multiple 3D datasets found {datasets}. "
                        f"Specify one with --key (use full path inside mat)."
                    )
                key_use = datasets[0]
            else:
                key_use = key
                if key_use not in f:
                    raise KeyError(f"{mat_path.name}: dataset '{key_use}' not found in v7.3 file.")

            ds = f[key_use]
            arr = np.array(ds)  # loads this part into RAM (one .mat at a time)
            return arr

    except Exception as e:
        if scipy_error is not None:
            raise RuntimeError(
                f"Failed to load {mat_path.name} with scipy ({scipy_error}) and h5py ({e})."
            ) from e
        raise


def parse_meta(meta_path: Path) -> Tuple[int, int, int]:
    meta = json.loads(meta_path.read_text())
    no_frames_total = int(meta["no_frames_total"])
    no_parts = int(meta["no_parts"])
    no_frames = int(meta["no_frames"])
    return no_frames_total, no_parts, no_frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=str, required=True, help="Folder containing .mat files")
    ap.add_argument("--meta", type=str, required=True, help="Path to metadata JSON")
    ap.add_argument("--out", type=str, required=True, help="Output .zarr path (directory store)")
    ap.add_argument(
        "--key",
        type=str,
        default=None,
        help="Variable name in .mat (or dataset path in v7.3). If omitted, expects exactly one 3D array.",
    )
    ap.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Optional output dtype (e.g., uint8, uint16, float32). Default: preserve input dtype.",
    )
    ap.add_argument(
        "--chunks",
        type=int,
        nargs=3,
        default=(128, 512, 512),
        metavar=("CT", "CH", "CW"),
        help="Zarr chunk shape in (T,H,W). Default: 128 512 512",
    )
    ap.add_argument(
        "--clevel",
        type=int,
        default=5,
        help="Compression level for zstd (1-9 typical). Default: 5",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it exists (will delete the directory).",
    )
    args = ap.parse_args()

    # ---- Zarr v3 imports ----
    try:
        import zarr  # type: ignore
        # Zarr v3 uses "codecs". Blosc is provided by zarr via numcodecs/zarr-codecs plumbing.
        from zarr.codecs import BloscCodec  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Missing deps (or wrong zarr version). Install with:\n"
            "  pip install 'zarr>=3' scipy h5py\n"
            "If you also want Blosc, ensure numcodecs is installed:\n"
            "  pip install numcodecs\n"
        ) from e

    folder = Path(args.folder).expanduser()
    meta_path = Path(args.meta).expanduser()
    out_path = Path(args.out).expanduser()

    no_frames_total, no_parts, no_frames = parse_meta(meta_path)

    mat_files = sorted(folder.glob("*.mat"), key=lambda p: natural_key(p.name))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {folder}")

    if len(mat_files) != no_parts:
        raise ValueError(f"Metadata says no_parts={no_parts}, but found {len(mat_files)} .mat files.")

    # Infer dimensions from first file
    first = load_mat_array(mat_files[0], key=args.key)
    if first.ndim != 3:
        raise ValueError(f"{mat_files[0].name}: expected 3D array, got shape {first.shape}")

    H, W, T0 = first.shape  # WxHxT
    if T0 != no_frames:
        raise ValueError(f"{mat_files[0].name}: expected T={no_frames} from metadata, got {T0}")

    out_dtype = np.dtype(args.dtype) if args.dtype else first.dtype

    ct, ch, cw = map(int, args.chunks)

    # ---- Create output group (directory store) ----
    if out_path.exists():
        raise FileExistsError(
            f"Output path already exists: {out_path}. "
            f"Remove it or pass --overwrite."
        )

    # Zarr v3: open_group on a filesystem path creates a directory-backed store
    root = zarr.open_group(out_path.as_posix(), mode="w")

    # Store metadata
    root.attrs.update(
        {
            "source_folder": folder.as_posix(),
            "no_frames_total": int(no_frames_total),
            "no_parts": int(no_parts),
            "no_frames_per_part": int(no_frames),
            "input_shape_per_part_HWT": (int(H), int(W), int(no_frames)),
            "output_shape_THW": (int(no_frames_total), int(H), int(W)),
            "zarr_version": getattr(zarr, "__version__", "unknown"),
        }
    )

    if ch != H or cw != W:
        print(
            f"Note: chunks set to (T,H,W)=({ct},{ch},{cw}) while output is "
            f"(T,H,W)=({no_frames_total},{H},{W}). This is allowed."
        )

    # ---- Compression (Zarr v3) ----
    # Blosc with zstd is a good default.
    # shuffle='bitshuffle' helps for integer-ish scientific data.
    codec = BloscCodec(cname="zstd", clevel=int(args.clevel), shuffle="bitshuffle")

    # Create the array. In Zarr v3 this is create_array (preferred).
    z = root.create_array(
        name="data",
        shape=(no_frames_total, H, W),   # TxHxW
        chunks=(ct, ch, cw),
        dtype=out_dtype,
        compressors=(codec,),            # tuple of codecs (v3 pipeline)
        overwrite=False,
    )

    # Write loop
    write_pos = 0
    print("Order:")
    for p in mat_files:
        print(" ", p.name)

    for idx, mat_path in enumerate(mat_files):
        arr = load_mat_array(mat_path, key=args.key)
        if arr.shape != (H, W, no_frames):
            raise ValueError(
                f"{mat_path.name}: shape mismatch. Expected {(H, W, no_frames)}, got {arr.shape}"
            )

        if arr.dtype != out_dtype:
            arr = arr.astype(out_dtype, copy=False)

        # WxHxT -> TxHxW
        arr_thw = np.transpose(arr, (2, 0, 1))  # (T,H,W) for HxWxT

        end = write_pos + arr_thw.shape[0]
        if end > no_frames_total:
            raise ValueError(
                f"Would write past no_frames_total={no_frames_total}. "
                f"At file {idx} ({mat_path.name}), write_pos={write_pos}, end={end}."
            )

        z[write_pos:end, :, :] = arr_thw
        write_pos = end
        print(
            f"[{idx+1}/{len(mat_files)}] wrote {arr_thw.shape[0]} frames from {mat_path.name} "
            f"-> data[{write_pos}/{no_frames_total}]"
        )

    if write_pos != no_frames_total:
        raise ValueError(f"Finished writing {write_pos} frames, but metadata no_frames_total={no_frames_total}.")

    print(f"Done. Saved Zarr group: {out_path}")
    print(f"Dataset: data, shape={z.shape}, chunks={z.chunks}, dtype={z.dtype}")
    print(f"Compression: {codec}")


if __name__ == "__main__":
    main()