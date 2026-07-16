import argparse
import yaml
from uwb3d import motion_detection
from uwb3d.ihpp_data import read_quanta_zarr, to_video, read_spadbin, binframes_to_spt , thin_events_uniform, events_to_binframes
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import os
import logging
import cv2
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML file",
    )
    parser.add_argument(
        "-c", "--compensate_motion",
        action="store_true",
        help="Enable motion compensation",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging (INFO). Default is WARNING.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Optional path to a log file. If set, full INFO logs are also saved there.",
    )

    parser.add_argument(
        "-t", "--thin",
        type=int,
        default=1,
        help="Optional thinning factor: keep 1/p photons. Default is 1 (no thinning)."
    )

    parser.add_argument(
        "-d", "--detection_cuts",
        type=float,
        nargs=4,
        metavar=("vx_min", "vx_max", "vy_min", "vy_max"),
        help="Velocity crop: vx_min vx_max vy_min vy_max"
    )

    parser.add_argument(
        "-m", "--marker_size",
        type=int,
        default=60,
        help="Marker size for detections in the visualization. Default is 60."
    )

    return parser.parse_args()


def setup_logger(verbose: bool, log_file: str | None):
    """
    Console:
      - verbose=False -> WARNING+
      - verbose=True  -> INFO+
    File (if provided):
      - always INFO+
    """         
    logger = logging.getLogger("motion_run")
    logger.setLevel(logging.INFO)  # overall logger accepts INFO; handlers filter
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_level = logging.INFO if verbose else logging.WARNING
    sh = logging.StreamHandler()
    sh.setLevel(console_level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def fmt_path(s: str, fmt: dict) -> str:
    return s.format(**fmt)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_compensated_png_cv2(img: np.ndarray, out_png_path: str, log):
    """
    Save a single image as PNG using OpenCV (no axes, no matplotlib).
    Handles float images by normalizing to uint8.
    """
    if img.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape={img.shape}")

    # Convert to uint8 safely
    if img.dtype == np.uint8:
        out_u8 = img
    else:
        img_f = img.astype(np.float32)
        mn = float(np.nanmin(img_f))
        mx = float(np.nanmax(img_f))
        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
            out_u8 = np.zeros(img.shape, dtype=np.uint8)
        else:
            out_u8 = ((img_f - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)

    ok = cv2.imwrite(out_png_path, out_u8)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for: {out_png_path}")
    log.info("Saved compensated PNG (cv2) -> %s", out_png_path)

def sinebow_segment(
    n=256,
    gamma=1.0,
    start=0.0,
    end=0.5,
):
    """
    Use only a slice of sinebow between [start, end] in color-wheel units.

    start, end in [0,1]
        0.0–1.0 covers full cycle
        0.55–0.85 gives blue→yellow-ish range (nice!)

    """
    t = np.linspace(start, end, n, endpoint=True)

    r = np.sin(np.pi * t) ** 2
    g = np.sin(np.pi * (t + 1/3)) ** 2
    b = np.sin(np.pi * (t + 2/3)) ** 2

    rgb = np.stack([r, g, b], axis=1)

    if gamma != 1.0:
        rgb = rgb ** gamma

    return ListedColormap(rgb, name="sinebow_segment")

def main():
    args = parse_args()
    log = setup_logger(args.verbose, args.log_file)

    # ================================ Loading the data and configuration ================================ #

    log.info("Loading config: %s", args.config)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    marker_size = args.marker_size
    cuts_raw = args.detection_cuts
    if cuts_raw is not None:
        detection_cuts = ((cuts_raw[0], cuts_raw[1]), (cuts_raw[2], cuts_raw[3]))
        log.warning("Using detection cuts: vx in %s, vy in %s", detection_cuts[0], detection_cuts[1])
    else:
        log.warning("No detection cuts provided; showing full bounds.")
    thinning_factor = args.thin
    T_exp = cfg["data"]["T_exp"]
    type = cfg["data"].get("type", "zarr")
    num_frames = cfg["data"]["frames"]
    X_res = cfg["data"]["width"]
    Y_res = cfg["data"]["height"]
    slice_start, slice_end = cfg["data"]["slice"]
    frame_slice = slice(slice_start, slice_end)
    bin_path = cfg["data"]["path"]
    bounds = cfg["data"]["velocity_bounds"]

    window_size = cfg["window_settings"]["window_size"]
    window_stride = cfg["window_settings"]["window_stride"]
    temporal_size = cfg["window_settings"]["temporal_size"]
    patch_num = cfg["window_settings"]["patch_num"]

    vmin, vmax, n = bounds  # bounds = [vmin, vmax, n]
    dt = T_exp / num_frames
    sinebow_cmap = sinebow_segment(n=256, gamma=0.5, start=0.7, end=1.3)

    fmt = {
        "experiment_name": cfg.get("experiment_name", "exp"),
        "X_res": X_res,
        "Y_res": Y_res,
        "width": X_res,
        "height": Y_res,
        "velocity_bounds": bounds,
        "vmin": vmin,
        "vmax": vmax,
        "n_vel": n,
        "num_frames": num_frames,
        "slice_start": slice_start,
        "slice_end": slice_end,
        "window_size": window_size,
        "window_stride": window_stride,
        "temporal_size": temporal_size,
        "patch_num": patch_num,
    }

    base_dir = fmt_path(cfg["output"]["base_path"], fmt)
    ensure_dir(base_dir)

    sub_folder = fmt_path(cfg["output"]["sub_folder"], fmt)
    sub_folder += f"_thinned_{thinning_factor}"
    output_dir = os.path.join(base_dir, sub_folder)
    ensure_dir(output_dir)

    output_bin_video_path = os.path.join(output_dir, fmt_path(cfg["output"]["bin_video_path"], fmt))
    two_dimensional_energies_path = os.path.join(output_dir, fmt_path(cfg["output"]["two_dimensional_energies_path"], fmt))
    two_dimensional_detections_path = os.path.join(output_dir, fmt_path(cfg["output"]["two_dimensional_detections_path"], fmt))
    two_dimensional_motion_compensated_dir = os.path.join(
        output_dir, fmt_path(cfg["output"]["two_dimensional_motion_compensated_path"], fmt)
    )
    os.makedirs(two_dimensional_motion_compensated_dir, exist_ok=True)
    log.info("Motion-compensation outputs dir: %s", two_dimensional_motion_compensated_dir)


    fps = int(1 / (T_exp / num_frames))

    # Important summary -> WARNING so it appears without -v
    log.warning("Output directory: %s", output_dir)
    log.warning("Frames=%d, Res=%dx%d, FPS=%d, Slice=[%d,%d)", num_frames, X_res, Y_res, fps, slice_start, slice_end)
    log.info("Velocity bounds=%s (vmin=%s vmax=%s n=%s)", bounds, vmin, vmax, n)
    log.info("Window size=%d stride=%d temporal=%d patch_num=%d", window_size, window_stride, temporal_size, patch_num)
    log.info("dt=%g seconds/frame", dt)

    # ================================ Load data ================================ #

    log.info("Loading the binary video from %s", bin_path)
    if type == "zarr":
        spadbinframes, _, _ = read_quanta_zarr(bin_path)
    else:
        spadbinframes = read_spadbin(bin_path,nframes=slice_end,h=Y_res,w=X_res,downsample=int(512 // X_res),keep_prob=1.0,hotpix_thresh=3.0)
    log.info("Loaded spadbinframes shape: %s", spadbinframes.shape)
    # ================================ Patch extraction ================================ #

    analyzed_spadbinframes = spadbinframes[frame_slice, :, :]
    log.info("After frame slice: %s", analyzed_spadbinframes.shape)

    if window_size > X_res or window_size > Y_res:
        log.warning("Window size %d is larger than frame dimensions (%dx%d). Adjusting to fit.", window_size, X_res, Y_res)
        window_size = (min(window_size, Y_res),min(window_size, X_res))

    patches_list = motion_detection.extract_patches_3d(
        analyzed_spadbinframes,
        patch_size=window_size,
        stride=window_stride,
        temporal_size=temporal_size,
    )
    log.info("Number of patches: %d", len(patches_list))

    windows = np.array([patch for _, _, patch in patches_list])
    analyzed_spadbinframes = windows[patch_num]

    effective_t_exp = T_exp * (analyzed_spadbinframes.shape[0] / num_frames)
    log.info("Selected patch shape: %s", analyzed_spadbinframes.shape)
    log.info(f"Effective exposure time: {effective_t_exp} seconds")

    # ================================ [Optional Thinning] ================================ #
    
    if thinning_factor > 1:
        log.info("Thinning binary frames by factor %d", thinning_factor)
        events = binframes_to_spt(analyzed_spadbinframes,effective_t_exp)
        events = np.array(events)
        events = np.transpose(events)
        log.info(f"Original events shape: {events.shape}")
        events_thinned = thin_events_uniform(events, 1/thinning_factor)
        log.info(f"Thinned events shape: {events_thinned.shape}")
        t,y,x = events_thinned[:,0], events_thinned[:,1], events_thinned[:,2]
        log.info(f"t.shape={t.shape} y.shape={y.shape} x.shape={x.shape}")
        analyzed_spadbinframes = events_to_binframes(x,y,t,analyzed_spadbinframes.shape[0], analyzed_spadbinframes.shape[1], analyzed_spadbinframes.shape[2])
        log.info("Converted back to binframes shape: %s", analyzed_spadbinframes.shape)
    else:
        log.info(f"No thinning applied (thin={thinning_factor})")

    # ================================ Save binary video ================================ #

    if not os.path.exists(output_bin_video_path):
        log.info("Saving binary patch video -> %s", output_bin_video_path)
        to_video(analyzed_spadbinframes, output_bin_video_path, 1, 30, "gray", "mp4")
    else:
        log.info("Video already exists: %s", output_bin_video_path)

    # ================================ FFT ================================ #

    log.info("Computing FFT (unshifted).")
    F, ft, fy, fx = motion_detection.fft3_unshifted(analyzed_spadbinframes, dt)

    F_shift = np.fft.fftshift(F)
    ft_shift = np.fft.fftshift(ft)
    fy_shift = np.fft.fftshift(fy)
    fx_shift = np.fft.fftshift(fx)

    log.info("FFT shapes: F=%s ft=%s fy=%s fx=%s", F.shape, ft.shape, fy.shape, fx.shape)

    # ================================ Energies ================================ #

    log.info("Computing energies in velocity planes (torch batched).")
    energies = motion_detection.compute_energies_velocity_planes_torch_batched(
        velocities_px_frame=bounds,
        ft=ft_shift,
        fy=fy_shift,
        fx=fx_shift,
        F=F_shift,
        num_t=analyzed_spadbinframes.shape[0],
        num_x=analyzed_spadbinframes.shape[1],
        num_y=analyzed_spadbinframes.shape[2],
        dc_bins=1,
        baseline_percentile=10.0,
        sigma=0.3,
        device="cuda",
        batch_size=10,
    )

    if not os.path.exists(two_dimensional_energies_path):
        log.info("Saving energies figure -> %s", two_dimensional_energies_path)
        fig = motion_detection.visualize_energies_on_multiple_component_velocity_planes(energies, bounds)
        fig.savefig(two_dimensional_energies_path, bbox_inches="tight", format="svg")
        plt.close(fig)
    else:
        log.info("Energies illustration already exists: %s", two_dimensional_energies_path)

    # ================================ Detection ================================ #

    log.info("Running CFAR detection from energies.")
    _, E_bg, det, thr = motion_detection.compute_cfar_rs_2d(
        energies, alpha=1e-3, window=30, guard=1
    )

    # ---- Save detection plot (still needs energies+det) ----
    if not os.path.exists(two_dimensional_detections_path):
        log.info("Saving detections figure -> %s", two_dimensional_detections_path)
        fig = motion_detection.visualize_cfar_statistics_2d(energies=energies, det=det, bounds=bounds, cmap=sinebow_cmap, marker_size=marker_size, cuts=detection_cuts)
        fig.savefig(two_dimensional_detections_path, bbox_inches="tight", format="svg")
        plt.close(fig)
    else:
        log.info("Detections illustration already exists: %s", two_dimensional_detections_path)

    # ---- fraction (torch-safe) ----
    try:
        fraction = (det.float().mean()).item()   # if det is torch
    except Exception:
        fraction = det.sum() / det.size          # if det is numpy

    log.warning("Detection fraction: %g", fraction)

    # ---- IMPORTANT: move det to CPU BEFORE freeing GPU stuff -----

    if isinstance(det, torch.Tensor):
        det_cpu = det.detach().cpu().numpy()
    else:
        det_cpu = det

    # ---- Build vx/vy values from CPU mask ----
    values = np.linspace(*bounds)
    vx_grid, vy_grid = np.meshgrid(values, values, indexing="ij")
    vx_values = vx_grid[det_cpu]
    vy_values = vy_grid[det_cpu]

    # ---- NOW it's safe to drop big GPU tensors ----
    # (do this BEFORE motion compensation)
    if torch.cuda.is_available():
        # if you won't need energies/E_bg/thr on GPU anymore:
        del energies, E_bg, det, thr
        torch.cuda.empty_cache()

    # ---- Logging list (CPU) ----
    if vx_values.size == 0:
        log.info("No detections to list (det mask is empty).")
    else:
        log.info("Listing %d detected (vx, vy) pairs:", vx_values.size)
        for vx_i, vy_i in zip(vx_values, vy_values):
            log.info("vx=% .4f  vy=% .4f", vx_i, vy_i)

    

    # =============================== Motion compensation ================================ #

    if not args.compensate_motion:
        log.info("Skipping motion compensation (flag not set).")
        return

    if vx_values.size == 0:
        log.warning("Motion compensation requested, but there are no detected velocities.")
        return


    # ================= CUDA CHECK =================

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        log.info("CUDA available — using torch GPU motion compensation.")
        device = torch.device("cuda")

        # Move spectral data once to GPU
        C_t  = torch.as_tensor(F,  device=device, dtype=torch.complex64)
        ft_t = torch.as_tensor(ft, device=device, dtype=torch.float32)
        fy_t = torch.as_tensor(fy, device=device, dtype=torch.float32)
        fx_t = torch.as_tensor(fx, device=device, dtype=torch.float32)
    else:
        log.info("CUDA not available — using NumPy CPU motion compensation.")


    log.info("Running motion compensation loop for every detected velocity.")

    for vx_i, vy_i in zip(vx_values, vy_values):
        vx_str = f"{vx_i:+.4f}".replace(".", "p")
        vy_str = f"{vy_i:+.4f}".replace(".", "p")

        log.info("Compensating for velocity (vx=% .4f, vy=% .4f)", vx_i, vy_i)

        # ================= COMPUTE (GPU or CPU) =================

        if use_gpu:
            img_stab2 = motion_detection.compensate_and_stabilize_tiled_torch(
                C_t, ft_t, fy_t, fx_t,
                vx=float(vx_i), vy=float(vy_i),
                tile_y=128, tile_x=128,   # tune if needed
            ).detach().cpu().numpy()
        else:
            C_desc = motion_detection.compensate_motion(F, ft, fy, fx, vx_i, vy_i)
            img_stab2 = motion_detection.stabilize_image_from_Fourier(C_desc, ft)

        # ================= SAVE PNG ONLY IMAGE =================

        only_png_name = f"vx_{vx_str}__vy_{vy_str}__ONLY_COMPENSATED.png"
        only_png_path = os.path.join(two_dimensional_motion_compensated_dir, only_png_name)

        if not os.path.exists(only_png_path):
            save_compensated_png_cv2(img_stab2, only_png_path, log)
        else:
            log.info("ONLY_COMPENSATED PNG already exists: %s", only_png_path)

        # ================= SAVE COMPARISON FIG =================

        compare_svg_name = f"vx_{vx_str}__vy_{vy_str}__COMPARISON.svg"
        compare_svg_path = os.path.join(two_dimensional_motion_compensated_dir, compare_svg_name)

        if not os.path.exists(compare_svg_path):
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            ax[0].imshow(np.sum(analyzed_spadbinframes, axis=0), cmap="gray")
            ax[0].axis("off")
            ax[0].set_title(f"Original Patch Sum\nvx={vx_i:.4f}, vy={vy_i:.4f}")

            ax[1].imshow(img_stab2, cmap="gray")
            ax[1].axis("off")
            ax[1].set_title(f"Compensated\nvx={vx_i:.4f}, vy={vy_i:.4f}")

            fig.savefig(compare_svg_path, bbox_inches="tight", format="svg")
            plt.close(fig)

            log.info("Saved comparison SVG -> %s", compare_svg_path)
        else:
            log.info("Comparison SVG already exists: %s", compare_svg_path)

        # ================= OPTIONAL DISPLAY =================

        if args.verbose:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(np.sum(analyzed_spadbinframes, axis=0), cmap="gray")
            ax[0].axis("off")
            ax[0].set_title(f"Original Patch Sum\nvx={vx_i:.4f}, vy={vy_i:.4f}")

            ax[1].imshow(img_stab2, cmap="gray")
            ax[1].axis("off")
            ax[1].set_title(f"Compensated\nvx={vx_i:.4f}, vy={vy_i:.4f}")

            plt.tight_layout()
            plt.show()
            plt.close(fig)


if __name__ == "__main__":
    main()