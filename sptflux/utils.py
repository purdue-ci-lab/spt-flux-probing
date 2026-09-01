"""
Miscellaneous utility functions.

Most general-purpose helpers now live in spadlib (spadlib.utils,
spadlib.io, spadlib.video_io). Only functions with no spadlib
equivalent remain here.
"""
import atexit
import datetime
import os
from pathlib import Path
import shutil
import tempfile
import weakref

import numpy as np
import sparse
import zarr

from spadlib.utils import is_json_serializable


class TempMemmap:
    """
    Temporary np.memmap wrapper that cleans up automatically.
    """

    def __init__(self, shape, dtype=np.float32, mode='w+'):
        # Create a real temporary filename that np.memmap can use
        tmp = tempfile.NamedTemporaryFile(delete=False)
        self.filename = tmp.name
        tmp.close()  # close file handle so memmap can reopen it

        # Create the memory-mapped array
        self.data = np.memmap(self.filename, dtype=dtype, mode=mode, shape=shape)

        # Register cleanup handlers
        atexit.register(self._cleanup)
        # weakref ensures cleanup even if the object is deleted before interpreter exit
        self._finalizer = weakref.finalize(self, TempMemmap._delete_file, self.filename)

    def _cleanup(self):
        """Flush and delete on normal interpreter exit."""
        if hasattr(self, "data"):
            try:
                self.data.flush()
                self.data._mmap.close()
            except Exception:
                pass
        TempMemmap._delete_file(self.filename)

    @staticmethod
    def _delete_file(fname):
        """Safely delete file if it exists."""
        try:
            if os.path.exists(fname):
                os.remove(fname)
        except Exception:
            pass

    def close(self):
        """Manually close and remove file."""
        self._cleanup()
        self._finalizer.detach()  # prevent double cleanup

    def __enter__(self):
        return self.data

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_path_stem(path: str | Path):
    """
    Path().stem except it checks zarr too to ignore the extension.
    Since zarr is a directory, and for all other directories you don't
    usually want to drop suffixes.
    """
    path = Path(path)
    if path.is_file():
        return path.stem
    if path.is_dir() and path.suffix == ".zarr":
        return path.stem
    return path.name


def zip_run(results_dir: str, out_dir: str, run_id: str | None = None, append_time=True) -> Path:
    """
    Zip up a results folder.
    """
    results_dir = Path(results_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = run_id or results_dir.name
    if append_time:
        base_name = f"{run_id}-{ts}"
    else:
        base_name = run_id
    final_zip = out_dir / f"{base_name}.zip"
    partial_zip = out_dir / f"{base_name}.zip.partial"

    # create archive in a temp location first
    with tempfile.TemporaryDirectory(dir=str(out_dir)) as td:
        tmp_zip_base = Path(td) / base_name  # shutil.make_archive wants a base path w/o extension
        tmp_zip_path = Path(shutil.make_archive(str(tmp_zip_base), "zip", root_dir=str(results_dir)))

        # move into out_dir as .partial then atomically rename to .zip
        shutil.move(str(tmp_zip_path), str(partial_zip))
        os.replace(partial_zip, final_zip)

    return final_zip


def save_sparse_to_zarr(arr: sparse.COO, zarr_path) -> zarr.Group:
    g = zarr.open_group(zarr_path, mode="w")

    g.create_array("coords", data=arr.coords, chunks="auto")
    g.create_array("data", data=arr.data, chunks="auto")
    g.attrs["shape"] = arr.shape
    if is_json_serializable(arr.fill_value):
        # too lazy to serialize complex numbers xd
        g.attrs["fill_value"] = arr.fill_value
    return g


def read_sparse_from_zarr(zarr_path) -> sparse.COO:
    g = zarr.open_group(zarr_path, mode="r")
    coords = g["coords"][:]
    data = g["data"][:]
    shape = tuple(g.attrs["shape"])
    fill_value = g.attrs.get("fill_value", 0)
    arr = sparse.COO(coords, data, shape=shape, fill_value=fill_value)
    return arr


def chunk_indices(sorted_arr, chunk_size, max_range, offset=0):
    """
    Partition a sorted 1D array into chunks and return a list of (start, end) index pairs
    (end is exclusive). Each chunk will contain at most `chunk_size` elements and will
    satisfy arr[end-1] - arr[start] <= max_range.

    Parameters
        sorted_arr: 1D array-like (must be sorted non-decreasing)
        chunk_size: positive integer, maximum number of elements per chunk
        max_range: non-negative number, maximum allowed value range within a chunk

    Returns
        List of (start, end) tuples where 0 <= start < end <= len(sorted_arr)
    """
    arr = np.asarray(sorted_arr)
    if arr.ndim != 1:
        raise ValueError("sorted_arr must be a 1D array-like")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if max_range < 0:
        raise ValueError("max_range must be non-negative")

    n = arr.shape[0]
    indices = []
    start = 0
    while start < n:
        # furthest index allowed by value range (first index > max_val)
        max_val = arr[start] + max_range
        pos = np.searchsorted(arr, max_val, side="right")
        # enforce chunk size and range constraint
        end = min(start + chunk_size, pos, n - 1)
        # ensure at least one element is taken (covers edge cases)
        if end == start:
            end = start + 1
        indices.append((start + offset, end + offset))
        start = end

    return indices


def _norm_slice(s: slice, n: int):
    """Return (start, stop, step) with Python's slice semantics."""
    start, stop, step = s.indices(n)
    if step <= 0:
        raise ValueError("Only positive step slices are supported here.")
    return start, stop, step


def map_chunk_to_compact_selection(
    *,
    chunk_start: int,
    chunk_stop: int,
    inner_start: int,
    inner_stop: int,
    sel: slice,
    n_global: int,
    sel_origin: int,
):
    """
    Intersect:
        [chunk_start, chunk_stop)  (step=1)
      ∩ [inner_start, inner_stop)  (step=1)
      ∩ sel (step=sel_step, in global coords)

    Returns:
        out_slc   : slice into compacted output axis (step=1)
        chunk_slc : slice into chunk-local axis (step=sel_step, relative to chunk_start)
    where compacted output axis is defined by indices in sel:
        out_index = (global_index - sel_start) // sel_step

    Parameters:
      sel_origin: the sel_start used to define the compacted output indexing.
                  Typically sel_origin == sel_start (after .indices()).
                  (Kept explicit so you don't accidentally use chunk_start.)
    """
    sel_start, sel_stop, sel_step = _norm_slice(sel, n_global)

    # First apply the step=1 constraints
    valid_start = max(chunk_start, inner_start)
    valid_stop  = min(chunk_stop,  inner_stop)
    if valid_start >= valid_stop:
        return slice(0, 0, 1), slice(0, 0, 1)

    # Now intersect with sel's bounds (still step=1 for the interval)
    overlap_start = max(valid_start, sel_start)
    overlap_stop  = min(valid_stop,  sel_stop)
    if overlap_start >= overlap_stop:
        return slice(0, 0, 1), slice(0, 0, 1)

    # Find the first index >= overlap_start that satisfies sel congruence:
    # global ≡ sel_start (mod sel_step)
    rem = (overlap_start - sel_start) % sel_step
    first = overlap_start if rem == 0 else overlap_start + (sel_step - rem)
    if first >= overlap_stop:
        return slice(0, 0, 1), slice(0, 0, 1)

    # Find the last index < overlap_stop satisfying the congruence
    last = overlap_stop - 1
    last -= (last - sel_start) % sel_step
    if last < first:
        return slice(0, 0, 1), slice(0, 0, 1)

    # Map to compact output coordinates (step 1)
    out_start = (first - sel_origin) // sel_step
    out_stop  = (last  - sel_origin) // sel_step + 1
    out_slc = slice(out_start, out_stop, 1)

    # Map to chunk-local coordinates (step sel_step)
    chunk_slc = slice(first - chunk_start, last - chunk_start + 1, sel_step)

    return out_slc, chunk_slc


def translate_slice_no_clamp(slc: slice, offset: int):
    """Translate without clamping (clamping is the caller's job)."""
    if offset == 0:
        return slc
    start = None if slc.start is None else slc.start + offset
    stop  = None if slc.stop  is None else slc.stop  + offset
    return slice(start, stop, slc.step)
