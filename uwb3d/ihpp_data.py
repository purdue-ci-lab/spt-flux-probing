"""
General data processing functions.

Most SPAD/quanta I/O, thinning, conversion, and flux utilities now live in
spadlib (spadlib.io, spadlib.processing, spadlib.sim). Only functions with no
spadlib equivalent remain here.
"""
import cv2
import numpy as np
from tqdm import tqdm

from spadlib.processing import correct_hotpixels_conv, thin_frames_uniform

from uwb3d.logger import get_logger


logger = get_logger(__name__)


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
