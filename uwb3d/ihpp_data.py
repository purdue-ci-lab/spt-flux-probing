"""
General data processing functions.
"""
from pathlib import Path
import re
from typing import Union
import warnings

import cv2
import dask.array as da
from matplotlib import cm
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
import time
from tqdm import tqdm
import zarr

from uwb3d import utils
from uwb3d.logger import get_logger

try:
    import cupy as cp
except ImportError:
    cp = None


logger = get_logger(__name__)


to_video = utils.to_video


def thin_events_uniform(events, keep_prob, dcr_rate=None, return_idx=False, seed=None, cuda=False):
    """
    Thin events uniformly with probability keep_prob.
    """
    xp = cp if cuda else np
    rng = xp.random.default_rng(seed=seed)
    mask = rng.random(events.shape[0]) < keep_prob
    if return_idx:
        return xp.nonzero(mask)
    return events[mask]


def thin_events_pixel_timeseries(pixel_timeseries, keep_prob, dcr_rate=None, seed=None, cuda=False):
    """
    Thin events uniformly with probability keep_prob, applied to a list of lists of pixel timeseries.

    Args:
        pixel_timeseries: list of lists of arrays, where pixel_timeseries[y][x] is an array of timestamps for that pixel.
        keep_prob: probability to keep each event.
        dcr_rate: if not None, will add dark count events with the given rate (in Hz) to each pixel after thinning.
    """
    xp = cp if cuda else np
    rng = xp.random.default_rng(seed=seed)
    thinned_pixel_timeseries = [[None] * len(row) for row in pixel_timeseries]
    pbar = tqdm(total=len(pixel_timeseries) * len(pixel_timeseries[0]), desc="Thinning events in pixels")
    for y in range(len(pixel_timeseries)):
        for x in range(len(pixel_timeseries[y])):
            if pixel_timeseries[y][x] is not None:
                events = pixel_timeseries[y][x]
                mask = rng.random(events.shape[0]) < keep_prob
                thinned_events = events[mask]
                if dcr_rate is not None:
                    dcr_prob = prob_from_dcr(dcr_rate, fps=1 / np.median(np.diff(events)))  # approximate fps from median inter-event time
                    n_dcr_events = rng.binomial(len(thinned_events), dcr_prob)
                    dcr_events = rng.uniform(0, events.max(), size=n_dcr_events)
                    thinned_events = xp.sort(xp.concatenate([thinned_events, dcr_events]))
                thinned_pixel_timeseries[y][x] = thinned_events
                pbar.update(1)
    return thinned_pixel_timeseries


def pixel_timeseries_to_npys(pixel_timeseries, output_dir):
    """
    Save a list of lists of pixel timeseries to .npy files, where pixel_timeseries[y][x] is an array of timestamps for that pixel.
    The files will be named with the pattern "posX{X}_posY{Y}.npy" where {X} and {Y} are the pixel coordinates.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pbar = tqdm(total=len(pixel_timeseries) * len(pixel_timeseries[0]), desc="Saving pixel timeseries to .npy files")
    for y in range(len(pixel_timeseries)):
        for x in range(len(pixel_timeseries[y])):
            if pixel_timeseries[y][x] is not None:
                np.save(output_dir / f"scan_posX{x:03d}_posY{y:03d}.npy", pixel_timeseries[y][x])
                pbar.update(1)

def prob_from_dcr(dcr_rate_hz, fps):
    """
    Convert a dark count rate in Hz to a per-frame probability of a dark count photon.
    """
    return 1 - np.exp(-dcr_rate_hz / fps)


def thin_frames_uniform(frames, p, dcr_prob=None, t_chunk=1000, seed=None):
    """
    Implementation from Andrew.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0,1].")
    if p == 1.0:
        return
    frames = np.copy(frames)  # avoid modifying in-place
    P = frames.shape[0]
    rng = np.random.default_rng(seed=seed)
    for t0 in tqdm(range(0, P, t_chunk), desc="Thinning frames in chunks"):
        t1 = min(P, t0 + t_chunk)
        slab = frames[t0:t1]  # (Bt,H,W) uint8
        U = rng.random(slab.shape)  # float64, ephemeral
        keep = (slab > 0) & (U < p)
        if dcr_prob is not None:
            dcr_mask = rng.random(slab.shape) < dcr_prob
            keep = keep | dcr_mask
        slab[...] = 0
        slab[keep] = 1
        frames[t0:t1] = slab
    return frames


def thin_frames_uniform_da(frames, keep_prob, dcr_prob=None, seed=None):
    """
    Thin binary frames uniformly with probability keep_prob. Also adds dark
    count photons to lower the SNR.

    This is an expensive operation on SPAD data, so dask is used for
    multiprocessing.

    Args:
        dcr_prob: dark count photon probability (not the rate itself)
    """
    T, H, W = frames.shape
    # convert to a dask array with automatic chunking and apply a lazy random mask
    frames = da.from_array(frames, chunks=(400, H, W))
    rs = da.random.RandomState(seed)
    mask = rs.random(frames.shape, chunks=frames.chunks) < keep_prob
    frames = (frames.astype("uint8") & mask.astype("uint8"))
    if dcr_prob is not None:
        dcr_photons = rs.binomial(1, dcr_prob, size=frames.shape, chunks=frames.chunks).astype("uint8")
        frames = frames | dcr_photons
    out = frames.compute()
    del frames
    return out


def thin_frames_counts_da(frames, keep_prob, dcr_rate=None, seed=None):
    """
    Thin photon-count frames by independently dropping each photon with
    probability (1 - keep_prob). Also adds dark count photons to lower the SNR.

    For each pixel with count k, the surviving count is drawn from
    Binomial(k, keep_prob). Dark counts are drawn from Poisson(dcr_rate).

    This is an expensive operation on SPAD data, so dask is used for
    multiprocessing.

    Args:
        frames:     Array of shape (T, H, W) with non-negative integer photon counts.
        keep_prob:  Probability that any individual photon survives thinning.
        dcr_rate:   Expected number of dark count photons added per pixel per frame.
                    (Poisson-distributed; pass None or 0 to skip.)
        seed:       Optional integer seed for reproducibility.
    """
    T, H, W = frames.shape
    frames = da.from_array(frames, chunks=(400, H, W))
    rs = da.random.RandomState(seed)

    # Each photon survives independently with keep_prob → Binomial thinning
    thinned = rs.binomial(frames, keep_prob, size=frames.shape, chunks=frames.chunks)

    # Dark counts: Poisson(dcr_rate) per pixel
    if dcr_rate is not None and dcr_rate > 0:
        dcr_photons = rs.poisson(dcr_rate, size=frames.shape, chunks=frames.chunks)
        thinned = thinned + dcr_photons

    out = thinned.compute()
    del thinned
    del frames
    return out


def write_quantaframes_zarr(path, frames, T_exp=None, fps=None, save_coords=False):
    """
    Write binary quanta frames to a Zarr group.

    Args:
        path (str or Path): Path to the output Zarr file.
        frames (np.ndarray): Array of shape (T, H, W) with binary frames.
        T_exp (float): exposure time in seconds.
        fps (float): if T_exp is None, fps must be provided to calculate T_exp.
        save_coords (bool): if True, also saves the (t, y, x) coordinates of events (for
            speed purposes so they don't have to be recomputed). Set to False to save
            disk space.
    """
    if T_exp is None:
        if fps is not None:
            T_exp = frames.shape[0] / fps
        else:
            raise ValueError("Either T_exp or fps must be provided")
    elif fps is not None and T_exp is not None:
        raise ValueError("Only one of T_exp or fps should be provided")
    elif T_exp is not None and fps is None:
        fps = frames.shape[0] / T_exp
    frames = np.asarray(frames, dtype="uint8")
    npoints = np.count_nonzero(frames)
    npts_persec_perpix = np.count_nonzero(frames, axis=0) / T_exp
    # statistics
    avg_pts_persec_perpixel = npoints / (T_exp * frames.shape[1] * frames.shape[2])
    max_pts_persec_perpixel = np.max(npts_persec_perpix)
    min_pts_persec_perpixel = np.min(npts_persec_perpix)
    med_pts_persec_perpixel = np.median(npts_persec_perpix)
    stddev_pts_persec_perpixel = np.std(npts_persec_perpix, ddof=1)
    fps = fps or (frames.shape[0] / T_exp)
    frame_size_bytes = frames.shape[1] * frames.shape[2]  # H * W for uint8
    max_chunk_bytes = 200_000_000
    max_frames_per_chunk = max(1, max_chunk_bytes // frame_size_bytes)
    if save_coords:
        t, y, x = binframes_to_spt(frames, T_exp=T_exp, normalize=True)
        quantadata = {
            "frames": frames,
            "t": t,
            "y": y,
            "x": x
        }
        chunks = {
            "frames": (max_frames_per_chunk, frames.shape[1], frames.shape[2]),
            "t": (100_000_000,),
            "y": (100_000_000,),
            "x": (100_000_000,),
        }
    else:
        quantadata = {
            "frames": frames
        }
        chunks = {
            "frames": (max_frames_per_chunk, frames.shape[1], frames.shape[2]),
        }
    compressors = {
        "frames": zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
    }
    utils.save_arrs_to_zarr(
        quantadata, path,
        chunks=chunks,
        n_workers=48,
        overwrite=True,
        attrs={
            "fps": fps,
            "T_exp": T_exp,
            "T": frames.shape[0],
            "H": frames.shape[1],
            "W": frames.shape[2],
            "shape": "(T, H, W)",
            "npoints": npoints,
            "avg_pts_persec_perpixel": avg_pts_persec_perpixel,
            "max_pts_persec_perpixel": max_pts_persec_perpixel,
            "min_pts_persec_perpixel": min_pts_persec_perpixel,
            "med_pts_persec_perpixel": med_pts_persec_perpixel,
            "stddev_pts_persec_perpixel": stddev_pts_persec_perpixel,
        },
        compressor_dict=compressors,
    )


def read_spadfolder(dirpath, h, w, T_exp, keep_prob=1.0, normalize=False):
    """
    Reads a folder of .npy files containing per-pixel timestamp arrays.
    Each .npy file is expected to be named with the pattern "posX{X}_posY{Y}.npy"
    where {X} and {Y} are the pixel coordinates.

    If there are multiple channels, there will be multiple folders. Run this for each
    folder.

    Returns:
        tuple:
            - points (np.ndarray): array of shape (N, 3) with columns [t, y, x].
            - pixel_timeseries (list of lists): list of lists of arrays.
    """
    dirpath = Path(dirpath)
    pixel_timeseries = [[None] * w for _ in range(h)]
    all_points = []
    for fn in tqdm(sorted(dirpath.glob("*.npy")), desc="Reading pixel timestamps from files"):
        m = re.search(r"posX(\d+)_posY(\d+)", fn.stem)
        if m is None:
            logger.debug(f"skipping file with unexpected name: {fn.name}")
            continue
        px = int(m.group(1))
        py = int(m.group(2))

        timestamps = np.load(fn)
        timestamps = timestamps[timestamps <= T_exp]
        if timestamps.size == 0:
            continue
        if normalize:
            timestamps /= T_exp  # normalize to [0, 1]

        if normalize:
            # normalize to [0, 1]
            x_vals = np.full(timestamps.shape[0], float(px) / w)
            y_vals = np.full(timestamps.shape[0], float(py) / h)
        else:
            x_vals = np.full(timestamps.shape[0], float(px))
            y_vals = np.full(timestamps.shape[0], float(py))

        pts = np.stack([
            timestamps.astype("float64"),
            y_vals,
            x_vals
        ], axis=1)
        all_points.append(pts)
        pixel_timeseries[py][px] = timestamps
    points = np.vstack(all_points)
    logger.info(f"Read {points.shape[0]} points from folder {dirpath}")
    if keep_prob < 1.0:
        nevents_orig = points.shape[0]
        points = thin_events_uniform(points, keep_prob=keep_prob, seed=42)
        logger.info(f"Kept {points.shape[0]} of {nevents_orig} points after thinning")
    return points, pixel_timeseries


def read_asyncspad_zarr(path, load_data=True):
    root = zarr.open_group(path, mode="r")
    
    # can't really lazily load this data
    timestamps = root["timestamps"][:]
    offsets = root["offsets"][:]
    lengths = root["lengths"][:]

    H, W = offsets.shape
    pixel_timeseries = [[None for _ in range(W)] for _ in range(H)]
    
    for i in range(H):
        for j in range(W):
            start = offsets[i, j]
            length = lengths[i, j]
            pixel_timeseries[i][j] = timestamps[start : start + length]
    t, y, x = root["t"], root["y"], root["x"]
    if load_data:
        t = t[:]
        y = y[:]
        x = x[:]
    return (t, y, x), pixel_timeseries, root.attrs


def read_spadbin(
    path, nframes, h, w, keep_prob=1.0, downsample=1, hotpix_thresh=None, hotpix_kernel_size=5,
    downsample_method="resize",
):
    """
    Reads a .bin file from a SPAD 512 camera and returns a numpy array of shape
    (nframes, h, w) containing the binary pixel data.

    There are probably hundreds of thousands of frames in the .bin, so set nframes
    accordingly.

    Args:
        hotpix_thresh (float or None): if not None, will do hot pixel correction based on
            the given stddev threshold and kernel size.
        downsample_method (str): "resize" to use cv2 resize, "slice" to use slicing.
    """
    nbytes_per_frame = h * w // 8
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(nframes * nbytes_per_frame), dtype=np.uint8)
    databit = np.unpackbits(data)
    wholetotalbit = databit.reshape((nframes, w, h))
    # transpose width/height and set origin to top-left
    wholetotalbit = wholetotalbit.transpose(0, 2, 1)[:, ::-1, :]
    if hotpix_thresh is not None:
        arr_hotpixel_corrected, hot_pixels, neighbor_mean = correct_hotpixels_conv(
            wholetotalbit, thresh_std=hotpix_thresh, kernel_size=hotpix_kernel_size,
        )
        nhotpixs = np.sum(hot_pixels)
        wholetotalbit = arr_hotpixel_corrected
        logger.info(f"Replaced {nhotpixs} hot pixels by neighbor comparison (threshold={hotpix_thresh})")
    if keep_prob < 1.0:
        nevents_orig = int(np.sum(wholetotalbit))
        wholetotalbit = thin_frames_uniform(wholetotalbit, keep_prob, seed=42)
        total_kept = int(np.sum(wholetotalbit))
        logger.info(f"Kept {total_kept} of {nevents_orig} points after thinning")
    if downsample > 1:
        if downsample_method == "resize":
            downsampled = []
            for frame in tqdm(wholetotalbit, desc="Downsampling frames"):
                downsampled.append(cv2.resize(frame, (w // downsample, h // downsample), interpolation=cv2.INTER_NEAREST))
            wholetotalbit = np.array(downsampled)
        elif downsample_method == "slice":
            wholetotalbit = wholetotalbit[:, ::int(downsample), ::int(downsample)]
    return wholetotalbit


def read_quanta_bin(path, H=512, W=512):
    """
    Read a single binary SPAD512 file.

    Ripped from spadtools (https://github.com/lyehe/spadtools).
    """
    if H is None:
        H = 512
    if W is None:
        W = 512
    with open(path, "rb") as f:
        raw_data = f.read()
    binframe_length = H * W // 8
    frame_count = len(raw_data) // binframe_length
    bits_array = np.frombuffer(raw_data, dtype=np.uint8).reshape(
        frame_count, H, W // 8
    )

    frames = np.unpackbits(bits_array, axis=2)
    frames = np.rot90(frames, k=1, axes=(1, 2))
    return frames


def read_quanta_dir(path, H=512, W=512):
    """
    Read a directory of binary SPAD512 files and concatenate them into a single array.
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Path {path} is not a directory")
    try:
        binpaths = sorted(path.glob("RAW*.bin"))
        all_frames = []
        for f in tqdm(binpaths, desc="Reading .bin files from directory"):
            all_frames.append(read_quanta_bin(f, H=H, W=W))
        all_frames = np.concatenate(all_frames, axis=0)
        return all_frames
    except ValueError as e:
        logger.error(f"Error with reading likely due to MacOS dotfiles or other garbage. Run `dot_clean` in the directory to clean up dotfiles.")
        raise e


def read_quanta_zarr(path, load_data=True):
    """
    Reads quanta data from a Zarr v3 group and returns:
      - frames: (T,H,W) array (lazy unless load_data=True)
      - (t,y,x): optional coordinate arrays if present
      - attrs: group attributes

    Works with:
      - "original" QuantaBurst zarr (frames/quantaframes + rich attrs)
      - your concatenated zarr (data + different attrs)
    """
    zarrdata = zarr.open_group(str(path), mode="r")

    # Zarr v3 safe listing of arrays
    keys = set(zarrdata.array_keys())

    # Accept multiple possible dataset names
    candidate_keys = ("frames", "quantaframes", "data")
    framekey = next((k for k in candidate_keys if k in keys), None)
    if framekey is None:
        raise KeyError(
            f"No frames array found in {path}. "
            f"Available arrays: {sorted(keys)}"
        )

    frames = zarrdata[framekey]

    # Optional coordinates
    coord_keys = set(zarrdata.array_keys())  # arrays again
    contains_coords = all(k in coord_keys for k in ("t", "y", "x"))
    if contains_coords:
        t = zarrdata["t"]
        y = zarrdata["y"]
        x = zarrdata["x"]
    else:
        t = y = x = None

    if load_data:
        frames = frames[:]
        if contains_coords:
            t = t[:]
            y = y[:]
            x = x[:]
            
    # attrs differ between your two formats; just return them as-is
    return frames, (t, y, x), dict(zarrdata.attrs)


def read_quanta_auto(path, load_data=True, H=None, W=None):
    """
    Automatically detects the format of the input path and reads the quanta data accordingly.
    Priority: .zarr > .bin > directory of .bin files.

    Returns:
        tuple:
            - quantaframes: array of shape (T, H, W) with binary frames.
            - metadata (dict or None): relevant metadata if available, else None.
    """
    path = Path(path)
    # first check if the path ends in .zarr, or if you append .zarr to the dir name it exists
    if path.suffix == ".zarr":
        if path.is_dir():
            logger.info(f"Reading quanta from Zarr file: {path}")
            frames, _, metadata = read_quanta_zarr(path, load_data=load_data)
            return frames, metadata
    elif (path.parent / f"{path.name}.zarr").is_dir():
        logger.info(f"Reading quanta from Zarr file: {path.parent / f'{path.name}.zarr'}")
        frames, _, metadata = read_quanta_zarr(path.parent / f"{path.name}.zarr", load_data=load_data)
        return frames, metadata
    # check if the path ends in .bin, or if you append .bin to the name it exists
    elif path.suffix == ".bin":
        if path.is_file():
            logger.info(f"Reading quanta from binary file: {path}")
            frames = read_quanta_bin(path, H=H, W=W)
            return frames, None
    elif (path.parent / f"{path.name}.bin").is_file():
        logger.info(f"Reading quanta from binary file: {path.parent / f'{path.name}.bin'}")
        frames = read_quanta_bin(path.parent / f"{path.name}.bin", H=H, W=W)
        return frames, None
    # otherwise, assume it's a directory of .bin files
    logger.info(f"Reading quanta from directory of binary files: {path}")
    frames = read_quanta_dir(path, H=H, W=W)
    return frames, None


def correct_hotpixels_conv(arr, thresh_std=5.0, kernel_size=5):
    """
    Hot pixel correction via convolutional neighbor statistics.
    """
    arr = arr.copy()
    # count events per pixel over the whole timespan
    pixel_event_counts = np.sum(arr, axis=0).astype(float)

    # neighborhood sums (including center)
    kernel = np.ones((kernel_size, kernel_size), dtype=float)
    total = cv2.filter2D(pixel_event_counts, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    # neighbors = total - center
    sum_neighbors = total - pixel_event_counts

    # how many neighbors each pixel has (edges will have fewer than nxn-1)
    neighbor_counts = cv2.filter2D(np.ones_like(pixel_event_counts, dtype=float), -1, kernel, borderType=cv2.BORDER_CONSTANT) - 1.0
    neighbor_counts_safe = np.where(neighbor_counts == 0, 1.0, neighbor_counts)

    # neighbor mean
    neighbor_mean = sum_neighbors / neighbor_counts_safe

    # neighbor std via mean of squares
    total_sq = cv2.filter2D(pixel_event_counts ** 2, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    sumsq_neighbors = total_sq - pixel_event_counts ** 2
    neighbor_mean_sq = sumsq_neighbors / neighbor_counts_safe
    neighbor_var = np.maximum(neighbor_mean_sq - neighbor_mean ** 2, 0.0)
    neighbor_std = np.sqrt(neighbor_var)

    # mark as hot if significantly above neighbors (hotpix_thresh acts as multiplier of std)
    hot_pixels = pixel_event_counts > (neighbor_mean + thresh_std * neighbor_std)
    # if neighbor_std is zero, mark only if strictly greater than neighbor mean
    hot_pixels = hot_pixels | ((neighbor_std == 0) & (pixel_event_counts > neighbor_mean))

    # zero out all hot pixels across time
    # arr is expected shape (T, H, W)
    try:
        arr[:, hot_pixels] = 0
    except Exception:
        # fallback: assign per-frame if boolean advanced indexing fails
        for t in range(arr.shape[0]):
            arr[t][hot_pixels] = 0

    # For each hot pixel, iterate over time and probabilistically revive events
    # so that pixels with more active neighbors have a higher chance of revival.
    # Probability is proportional to the number of active neighbors, capped at 1.0.
    rng = np.random.default_rng()
    nframes = arr.shape[0]
    for t in tqdm(range(nframes), total=nframes, desc="Reviving hot pixels for each frame"):
        frame = arr[t].astype(float)
        neigh_total = cv2.filter2D(frame, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        neigh_count = (neigh_total - frame)

        # probability of revival per pixel: fraction of neighbors that are active
        # (neighbor_counts_safe is from earlier; avoids division by zero)
        probs = neigh_count / neighbor_counts_safe
        probs = np.clip(probs, 0.0, 1.0)

        # draw random numbers and revive hot pixels according to probs
        rnd = rng.random(size=probs.shape)
        revive_mask = (rnd < probs) & hot_pixels

        if revive_mask.any():
            arr[t][revive_mask] = 1

    return arr, hot_pixels, neighbor_mean


def binframes_to_spt(frames: np.ndarray, T_exp=None, fps=None, keep_prob=1.0, normalize=True, cuda=False):
    """
    Convert (T, H, W) binary frames (0/1 per pixel) to spatiotemporal
    events (t, y, x).

    Args:
        frames (np.ndarray): array of shape (nframes, height, width) with
            binary pixel values.
        T_exp (float): exposure time in seconds.
        keep_prob (float): probability of keeping each event (for thinning).
        normalize (bool): if True, normalize x, y, t to [0, 1].

    Returns:
        tuple:
            - t: 1D array of time
            - y: 1D array of y coordinates
            - x: 1D array of x coordinates
    """
    xp = cp if cuda else np
    if T_exp is None:
        if fps is not None:
            T_exp = frames.shape[0] / fps
        else:
            raise ValueError("Either T_exp or fps must be provided")
    elif fps is not None and T_exp is not None:
        raise ValueError("Only one of T_exp or fps should be provided")
    # Get coordinates where pixel == 1
    t_coords, y_coords, x_coords = xp.nonzero(frames)
    t_coords = t_coords.astype("float64")
    y_coords = y_coords.astype("float64")
    x_coords = x_coords.astype("float64")

    # Stack and convert to float
    # x_coords += 0.5  # center of pixel x
    # y_coords += 0.5  # center of pixel y
    t_coords *= T_exp / frames.shape[0]  # convert frame index to time in seconds
    if keep_prob < 1.0:
        nevents_orig = t_coords.shape[0]
        idxs = thin_events_uniform(t_coords, return_idx=True, keep_prob=keep_prob, seed=42)
        t_coords = t_coords[idxs]
        y_coords = y_coords[idxs]
        x_coords = x_coords[idxs]
        logger.info(f"Kept {len(idxs)} of {nevents_orig} points after thinning")
    if normalize:
        # Normalize x, y, t to [0, 1]
        x_coords /= frames.shape[2]  # x
        y_coords /= frames.shape[1]  # y
        t_coords /= T_exp  # t
    return t_coords, y_coords, x_coords


def fluxfunc_from_frames(frames, conversion_gain=None, quant_eff=None, fps=None, normalize=True):
    """
    Convert frames discrete frames to a linearly interpolated continuous flux in photons/sec/pixel^2.
    
    Returns a function intensity(x, y, t) that linearly interpolates in x, y, t.

    If unnormalized, x and y are in pixel coordinates and t is in seconds. The flux would be
    in photons/sec/pixel^2.
    If normalized, x, y, t are in [0, 1], and the flux is also normalized by the dimensions so that
    the expected number of photons in the unit volume is the same as the expected number of photons in
    the physical volume.

    Args:
        frames (np.ndarray): array of intensity frames (T, H, W) or (T, H, W, 3) for RGB.
        conversion_gain (float): e per ADU
        quant_eff (float): quantum efficiency (0 to 1)
        fps (float): frames per second (for converting to photons/sec)
        normalize (bool): if True, flux x, y, t is evaluated on [0, 1] and scale flux accordingly.

    Returns:
        intensity function: function (x, y, t) -> float
        discr_flux: the discrete flux values on the original grid (T, H, W)
        max_flux: the maximum flux value
    """
    if len(frames.shape) == 4 and frames.shape[3] == 3:
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    else:
        gray_frames = [f for f in frames]
    pixvals = np.stack(gray_frames, axis=0).astype(np.float32)
    T, H, W = pixvals.shape
    flux_est = pixvals * conversion_gain * fps / quant_eff
    if normalize:
        # scale by the volume factor to preserve expected photon counts
        V = T * H * W
        flux_est *= V
    max_flux = np.max(flux_est)
    T_exp = frames.shape[0] / fps

    def intensity(x, y, t):
        # x, y, t can be floats or arrays
        if normalize:
            # [0, 1] for all coordinates
            coords = np.array([t * T, y * H, x * W])
        else:
            # physical units: seconds for t, pixels for x and y
            coords = np.array([t * fps, y, x])
        flux = map_coordinates(flux_est, coords, order=1, mode="nearest")
        return flux

    discr_flux = flux_est
    return intensity, discr_flux, max_flux


def flux_func_gaussian_ball(
    height, width, T_exp, x_dist=0.7, y_dist=0.7, hz_move_x=1, hz_move_y=0, hz_light=0, sigma=0.1, max_amplitude=1.0, normalize=True
):
    """
    Returns flux(x, y, t)
    """
    x_center = width / 2
    y_center = height / 2
    V = width * height * T_exp
    scale = 1
    if normalize:
        scale = V

    def flux(x, y, t):
        x = x.copy()
        y = y.copy()
        t = t.copy()
        if normalize:
            x *= width
            y *= height
            t *= T_exp
        curr_x_center = x_center + x_center * x_dist * np.cos(2 * np.pi * hz_move_x * t)
        curr_y_center = y_center + y_center * y_dist * np.sin(2 * np.pi * hz_move_y * t)
        curr_amplitudes = (0.5 + 0.5 * np.cos(2 * np.pi * hz_light * t)) * max_amplitude
        return scale * curr_amplitudes * np.exp(-((x - curr_x_center) ** 2 + (y - curr_y_center) ** 2) / (2 * sigma ** 2))
    return flux, max_amplitude * scale


def bin_spt_events(x, y, t, H=60, W=60):
    """
    Bin the spatiotemporal events by pixel, so each pixel contains a sorted
    timeseries of event times.

    x, y, t are assumed to be normalized to [0, 1].

    Returns:
        pixel_timeseries: 2D list of sorted event times for each pixel (so H x W
            list of arrays)
    """
    # Bin the spatiotemporal events by pixel, so each pixel contains a sorted timeseries of event times
    x_bins = np.linspace(0, 1, W + 1)
    y_bins = np.linspace(0, 1, H + 1)

    # Digitize x and y to get pixel indices
    x_idx = np.digitize(x, x_bins) - 1
    y_idx = np.digitize(y, y_bins) - 1

    # Ensure indices are within bounds
    x_idx = np.clip(x_idx, 0, W - 1)
    y_idx = np.clip(y_idx, 0, H - 1)

    # Prepare a 2D array of lists to hold event times for each pixel
    pixel_events = [[[] for _ in range(W)] for _ in range(H)]

    # Assign each event time to the corresponding pixel
    for xi, yi, ti in tqdm(zip(x_idx, y_idx, t), total=len(t), desc="Assigning events to pixels"):
        pixel_events[yi][xi].append(ti)

    # Convert lists to sorted numpy arrays for each pixel
    pixel_timeseries = [[np.sort(np.array(times)) for times in row] for row in pixel_events]
    return pixel_timeseries


def events_to_binframes(x, y, t, nframes, H, W, accumulate=False, cuda=False):
    """
    Convert spatiotemporal events (x, y, t) to binary frames of shape
    (nframes, H, W) where each pixel is 0/1.

    Args:
        x (np.ndarray): x coordinates in [0, 1].
        y (np.ndarray): y coordinates in [0, 1].
        t (np.ndarray): time coordinates in [0, 1].
        nframes (int): number of frames.
        H (int): height of each frame.
        W (int): width of each frame.

    Returns:
        frames (np.ndarray): binary frames of shape (nframes, H, W).
    """
    if cuda:
        frames = cp.zeros((nframes, H, W), dtype="uint8")
        # Convert normalized coordinates to pixel indices and frame indices
        x_idx = cp.clip((x * W).astype(int), 0, W - 1)
        y_idx = cp.clip((y * H).astype(int), 0, H - 1)
        t_idx = cp.clip((t * nframes).astype(int), 0, nframes - 1)

        if accumulate:
            frames = frames.astype("uint32")
            cp.add.at(frames, (t_idx, y_idx, x_idx), 1)
        else:
            # Set the corresponding pixels in the frames to 1
            frames[t_idx, y_idx, x_idx] = 1
    else:
        frames = np.zeros((nframes, H, W), dtype="uint8")
        # Convert normalized coordinates to pixel indices and frame indices
        x_idx = np.clip((x * W).astype(int), 0, W - 1)
        y_idx = np.clip((y * H).astype(int), 0, H - 1)
        t_idx = np.clip((t * nframes).astype(int), 0, nframes - 1)

        if accumulate:
            frames = frames.astype("uint32")
            np.add.at(frames, (t_idx, y_idx, x_idx), 1)
        else:
            # Set the corresponding pixels in the frames to 1
            frames[t_idx, y_idx, x_idx] = 1
    return frames


def timeseries_from_binframes(binframes, T_exp):
    """
    Converts a (T, H, W) binary frame array to a (H, W) list of sorted event times per
    pixel.
    """
    nframes, H, W = binframes.shape
    pixel_timeseries = [[[] for _ in range(W)] for _ in range(H)]

    pbar = tqdm(total=H * W, desc="Extracting event times per pixel")
    for y in range(H):
        for x in range(W):
            t_idx = np.nonzero(binframes[:, y, x])[0]
            if t_idx.size > 0:
                times = t_idx * (T_exp / nframes)
                pixel_timeseries[y][x] = times
            pbar.update(1)
    pbar.close()
    return pixel_timeseries


def spatial_chunk_slices(arr_or_shape, chunk_h=32, chunk_w=32):
    """
    Return list of (h_slice, w_slice) to split an image (or stack) into spatial chunks.
    Handles edge cases where H or W are not divisible by the chunk size: the last chunk
    simply ends at the image boundary and may be smaller.
    Accepts either:
      - a numpy array with shape (H, W) or (T, H, W) (T ignored for chunking),
      - or a shape tuple/list (H, W).
    """

    # determine H, W
    if hasattr(arr_or_shape, "shape"):
        shp = arr_or_shape.shape
    else:
        shp = tuple(arr_or_shape)

    if len(shp) == 2:
        H, W = shp
    elif len(shp) >= 3:
        # assume (T, H, W) or (C, H, W) etc. -> spatial dims are last two if needed,
        # but in this notebook convention is (T, H, W) so pick [-2], [-1] to be safe.
        H, W = shp[-2], shp[-1]
    else:
        raise ValueError("Expected 2D or 3D array/shape for spatial chunking")

    h_starts = list(range(0, H, chunk_h))
    w_starts = list(range(0, W, chunk_w))

    slices = []
    for hs in h_starts:
        he = min(hs + chunk_h, H)
        for ws in w_starts:
            we = min(ws + chunk_w, W)
            slices.append((slice(hs, he), slice(ws, we)))

    return slices


def iter_volume_chunks(arr, chunk_h=32, chunk_w=32, chunk_t=None, pad=0, pad_t=0):
    """
    Generator yielding ((tsl_or_None, hsl, wsl, chunk), (inner_t_slice_or_None, inner_h_slice, inner_w_slice, padded))
    for a given array `arr`.

    - If chunk_t is None (default) this behaves as before: tsl_or_None == None and inner_t_slice_or_None == None.
    - If chunk_t is provided (int > 0) and arr has shape (T, H, W), the function also chunks along time (first axis).
    - chunk is the original (unpadded) chunk: shape (chunk_t_actual, h, w) for temporal chunks or
      (h, w) / (..., h, w) for spatial-only.
    - padded contains neighborhood data around the chunk only for pixels/frames that exist inside the array
      (i.e. no zero-fill outside the bounds). Its size will be
      (t_top + chunk_t_actual + t_bottom, top + h + bottom, left + w + right) when temporal chunking is used,
      otherwise (top + h + bottom, left + w + right).
    - inner_*_slice can be used to extract `chunk` from `padded`:
        padded[inner_t, inner_h, inner_w] == chunk  (or with leading axes)
    """
    if pad < 0 or pad_t < 0:
        raise ValueError("pad and pad_t must be non-negative")

    if hasattr(arr, "shape"):
        shp = arr.shape
    else:
        raise ValueError("arr must be an array-like with shape")

    if chunk_t is None:
        chunk_t = shp[0]  # effectively no temporal chunking
    chunk_t = min(chunk_t, shp[0])  # cap chunk_t to T
    chunk_h = min(chunk_h, shp[1])  # cap chunk_h to H
    chunk_w = min(chunk_w, shp[2])  # cap chunk_w to W

    T, H, W = shp

    # spatial slices (as before)
    spatial_slices = spatial_chunk_slices((H, W), chunk_h=chunk_h, chunk_w=chunk_w)

    # temporal starts if requested
    t_starts = list(range(0, T, chunk_t))

    for ts in t_starts:
        chunk_t_actual = min(chunk_t, T - ts)
        te = (ts + chunk_t_actual)

        for hsl, wsl in spatial_slices:
            hs, he = hsl.start, hsl.stop
            ws, we = wsl.start, wsl.stop
            chunk_h_actual = he - hs
            chunk_w_actual = we - ws

            # extract core chunk
            chunk = arr[ts:te, hs:he, ws:we]

            # compute actual available spatial padding on each side (do not pad outside image)
            top_pad = min(pad, hs)
            bottom_pad = min(pad, H - he)
            left_pad = min(pad, ws)
            right_pad = min(pad, W - we)

            # compute available temporal padding if requested
            t_top_pad = min(pad_t, ts)
            t_bottom_pad = min(pad_t, T - te)

            # padded output dims (only includes available neighbors inside array)
            tgt_h_len = top_pad + chunk_h_actual + bottom_pad
            tgt_w_len = left_pad + chunk_w_actual + right_pad
            tgt_t_len = t_top_pad + chunk_t_actual + t_bottom_pad

            # inner slices to extract original chunk from the padded array
            inner_h_slice = slice(top_pad, top_pad + chunk_h_actual)
            inner_w_slice = slice(left_pad, left_pad + chunk_w_actual)
            inner_t_slice = slice(t_top_pad, t_top_pad + chunk_t_actual)

            # fast-path: if no padding available on any axis then padded == chunk
            no_padding = (top_pad == 0 and bottom_pad == 0 and left_pad == 0 and right_pad == 0 and
                          t_top_pad == 0 and t_bottom_pad == 0)
            if no_padding:
                padded = chunk.copy()
                tsl_out = slice(ts, te)
                yield ((tsl_out, hsl, wsl, chunk), (inner_t_slice, inner_h_slice, inner_w_slice, padded))
                continue

            # source region in the original array to copy into padded:
            src_h0 = hs - top_pad
            src_h1 = he + bottom_pad
            src_w0 = ws - left_pad
            src_w1 = we + right_pad

            src_t0 = ts - t_top_pad
            src_t1 = te + t_bottom_pad
            
            padded = np.zeros((tgt_t_len, tgt_h_len, tgt_w_len), dtype=arr.dtype)
            src = arr[src_t0:src_t1, src_h0:src_h1, src_w0:src_w1]
            padded[0:src.shape[0], 0:src.shape[1], 0:src.shape[2]] = src
            tsl_out = slice(ts, te)

            yield ((tsl_out, hsl, wsl, chunk), (inner_t_slice, inner_h_slice, inner_w_slice, padded))


class SPADHotpixelTool:
    """
    Ripped directly from bit2bit.

    Bro what is this class. They just put **kwargs everywhere and they separate
    the functions so much for hot/dead pixel correction you can barely follow it.

    Class to facilitate the hotpixel correction of SPAD data.

    :param data: 3D np array
    :param kwargs: Additional parameters
    """

    def __init__(self, data: np.ndarray, **kwargs):
        """Constructor method."""
        self.data = data
        self.flattend_data = self._flatten(self.data)
        if (zero_count := self._zero_count()) > 512:
            logger.warning(f"Possible insufficient data for correction. Zero count: {zero_count}")
        self.reset()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def reset(self, *args: str) -> "SPADHotpixelTool":
        """Reset the intermediate results of the hotpixel correction.

        :param args: Attributes to reset
        :return: The instance of the class
        """
        if not args:
            self.background = None
            self.hotpixel_image = None
            self.hotpixel_locations = None
            self.deadpixel_locations = None
            self.hotpixel_values = None
            self.expected_values = None
            self.probability_list = None
            self.corrected_image = None
        for arg in args:
            try:
                setattr(self, arg, None)
            except AttributeError:
                logger.error(f"Attribute {arg} not found")
        return self

    def _zero_count(self) -> int:
        """Return the number of zero in the flattened data."""
        count = np.sum(self.flattend_data < 1).item()
        return count

    @staticmethod
    def _flatten(data: np.ndarray) -> np.ndarray:
        """Reduce the dimension of the input data to 2D and save it as a np array."""
        if data.ndim > 2:
            data = np.sum(data, axis=tuple(range(data.ndim - 2)))
        return data.astype(np.float32)

    def get_background(self, method: str = "open", kernel_size: int = 3) -> np.ndarray:
        """Apply background filter to the input data and return the last result.

        :param method: Method for background filter, defaults to "open"
        :return: Data after background filter
        """
        image = self.flattend_data
        methods = {
            "open": lambda i, k: cv2.morphologyEx(
                i,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
            ),
            "mean": lambda i, k: cv2.blur(i, (k, k)),
            "median": lambda i, k: cv2.medianBlur(i.astype(np.float32), k),
            "median+mean": lambda i, k: methods["median"](methods["mean"](i, k), k),
        }
        if method in methods:
            self.background = methods[method](image, kernel_size)
        else:
            logger.error(f"Method {method} not implemented")
            raise NotImplementedError
        return self.background

    def get_hotpixels(self, kernel_size: int = 3) -> np.ndarray:
        """Subtract background from the input data to get hot pixels.

        :return: Data after hot pixel filter
        """
        if self.hotpixel_image is None:
            background_image = self.get_background(kernel_size=kernel_size)
            self.hotpixel_image = self.flattend_data - background_image
        return self.hotpixel_image

    def locate_hotpixels(
        self,
        hp_threshold: float = 2,
        **kwargs,
    ) -> np.ndarray:
        """Find hot pixels in the input data based on the standard deviation threshold.

        :param hp_threshold: Multiplyer for the standard deviation, defaults to 2
        :return: The locations of the hot pixels
        """
        if self.hotpixel_locations is None:
            hp_threshold = kwargs.get("hp_threshold", hp_threshold)
            hotpixel_image = self.get_hotpixels(**kwargs)
            mean = np.mean(hotpixel_image).item()
            std = np.std(hotpixel_image).item()
            hp_threshold = mean + hp_threshold * std
            self.hotpixel_locations = np.argwhere(hotpixel_image > hp_threshold)
        return self.hotpixel_locations

    def locate_deadpixels(self, **kwargs) -> np.ndarray:
        """Find dead pixels in the input data based on the standard deviation dp_threshold.

        :param dp_threshold: Multiplyer for the standard deviation, defaults to 1
        :return: The locations of the dead pixels
        """
        if self.deadpixel_locations is None:
            dp_threshold = kwargs.get("dp_threshold", 1)
            self.deadpixel_locations = np.argwhere(self.flattend_data < dp_threshold)
        return self.deadpixel_locations

    def correct_deadpixels(self, **kwargs) -> np.ndarray:
        """Correct the input data for dead pixels.

        :return: The corrected image
        """
        image = np.array(self.data) if self.corrected_image is None else self.corrected_image
        index = self.locate_deadpixels(**kwargs)
        kernel_size: int = kwargs.get("kernel_size", 3)
        background_image = self.get_background(kernel_size=kernel_size, method="median")
        expected_values = background_image[index[:, 0], index[:, 1]]
        for (x, y), value in zip(index, expected_values, strict=False):
            prob = value / image.shape[0]
            image[:, x, y] = np.random.binomial(1, prob, image.shape[0])
        self.corrected_image = image
        return self.corrected_image

    def get_hotpixel_values(self, **kwargs) -> np.ndarray:
        """Get the values of the hot pixels in the input data.

        :return: The values of the hot pixels
        """
        if self.hotpixel_values is None:
            image = self.flattend_data
            index = self.locate_hotpixels(**kwargs)
            self.hotpixel_values = image[index[:, 0], index[:, 1]]
        return self.hotpixel_values

    def get_expected_values(self, **kwargs) -> np.ndarray:
        """Get the values of the hot pixels in the background image.

        :return: The values of the hot pixels
        """
        if self.expected_values is None:
            kernel_size: int = kwargs.get("kernel_size", 3)
            background_image = self.get_background(kernel_size=kernel_size)
            index = self.locate_hotpixels(**kwargs)
            self.expected_values = background_image[index[:, 0], index[:, 1]]
        return self.expected_values

    def get_probablity(self, **kwargs) -> np.ndarray:
        """Get the probability of a hot pixel.

        :param probablity_base: Probability base, defaults to "hotpixels"
        :return: The probability of a hot pixel
        """
        if self.probability_list is None:
            probablity_base = kwargs.get("probablity_base", "hotpixels")
            expected_values = self.get_expected_values(**kwargs)
            if probablity_base == "all":
                self.probability_list = expected_values / self.data.shape[0]
            elif probablity_base == "hotpixels":
                hotpixel_values = self.get_hotpixel_values(**kwargs)
                self.probability_list = expected_values / hotpixel_values
            else:
                logger.error(f"Probability base {probablity_base} not implemented")
                raise NotImplementedError
        return self.probability_list

    def correct_hotpixels(self, probablity_base="hotpixels", hp_threshold: float=2.0, kernel_size: int=3) -> np.ndarray:
        """Correct the input data for hot pixels.

        :param probablity_base: Probability base, defaults to "hotpixels"
        :return: The corrected image
        """
        image = np.array(self.data) if self.corrected_image is None else self.corrected_image
        index = self.locate_hotpixels(hp_threshold=hp_threshold, kernel_size=kernel_size)
        p = self.get_probablity(probablity_base=probablity_base, hp_threshold=hp_threshold, kernel_size=kernel_size)
        n_iters = len(index)
        # should make np.nonzero faster... I think?
        image = np.asarray(image, dtype="bool")
        for (x, y), prob in tqdm(zip(index, p, strict=False), total=n_iters):
            if probablity_base == "all":
                filling = np.random.binomial(1, prob, image.shape[0])
                image[:, x, y] = filling
            elif probablity_base == "hotpixels":
                # z_idx = np.argwhere(image[:, x, y] == 1).flatten()
                z_idx = np.flatnonzero(image[:, x, y])
                image[z_idx, x, y] = np.random.binomial(1, prob, z_idx.size)
            else:
                logger.error(f"Probability base {probablity_base} not implemented")
        self.corrected_image = np.asarray(image, dtype="uint8")
        return self.corrected_image

    def inspect(self, **kwargs):
        """Show image results of the hotpixel correction."""
        try:
            import matplotlib.pyplot as plt

            deadpixel = kwargs.get("deadpixel", False)
            if deadpixel:
                self.correct_deadpixels(**kwargs)
            _, ax = plt.subplots(1, 2, figsize=(20, 40))
            ax[0].imshow(self.flattend_data, cmap="gray")
            ax[0].scatter(
                self.locate_hotpixels(**kwargs)[:, 1] + 3,
                self.locate_hotpixels(**kwargs)[:, 0],
                c="r",
                s=5,
                marker="_",
            )
            if deadpixel:
                ax[0].scatter(
                    self.locate_deadpixels(**kwargs)[:, 1] + 3,
                    self.locate_deadpixels(**kwargs)[:, 0],
                    c="g",
                    s=5,
                    marker="_",
                )
            title = f"Hotpixels (count: {self.locate_hotpixels(**kwargs).shape[0]})"
            if deadpixel:
                title += f" & Deadpixels (count: {self.locate_deadpixels(**kwargs).shape[0]})"
            ax[0].set_title(title)
            ax[0].set_axis_off()
            ax[1].imshow(np.sum(self.correct_hotpixels(**kwargs), axis=0), cmap="gray")
            ax[1].scatter(
                self.locate_hotpixels(**kwargs)[:, 1] + 3,
                self.locate_hotpixels(**kwargs)[:, 0],
                c="r",
                s=5,
                alpha=0.5,
                marker="_",
            )
            if deadpixel:
                ax[1].scatter(
                    self.locate_deadpixels(**kwargs)[:, 1] + 3,
                    self.locate_deadpixels(**kwargs)[:, 0],
                    c="g",
                    s=5,
                    alpha=0.5,
                    marker="_",
                )
            ax[1].set_title("Projection without hotpixels")
            ax[1].set_axis_off()
            plt.subplots_adjust(wspace=0.01, hspace=0)
            plt.show()
        except ImportError:
            logger.error("Matplotlib not installed")
