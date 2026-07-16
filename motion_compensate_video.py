import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
import torch
from tqdm.auto import tqdm

from sptflux import motion_detection
from sptflux.ihpp_data import read_spadbin
from spadlib.io import read_quanta_zarr, frames_to_events, events_to_frames
from spadlib.processing import thin_events_uniform


# -------------------------
# CLI + logging
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--vx", type=float, required=True, help="Compensation vx [px/frame]")
    p.add_argument("--vy", type=float, required=True, help="Compensation vy [px/frame]")

    p.add_argument(
        "-t",
        "--thin",
        type=int,
        default=1,
        help="Thinning factor: keep 1/thin events (default: 1 = no thinning)",
    )
    p.add_argument(
        "--skip",
        type=int,
        default=10,
        help="Frames to skip between compensations (default: 10)",
    )
    p.add_argument("--fps_out", type=int, default=30, help="Output video FPS (default: 30)")

    p.add_argument(
        "--window",
        type=int,
        default=None,
        help="Window length in frames (overrides window_settings.temporal_size)",
    )

    # NEW: logging knobs
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging (INFO). Default is WARNING.")
    p.add_argument("--log_file", type=str, default=None, help="Optional log file path (always INFO).")

    return p.parse_args()


def setup_logger(verbose: bool, log_file: str | None):
    logger = logging.getLogger("compensate_video")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO if verbose else logging.WARNING)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# -------------------------
# helpers
# -------------------------
def fmt_path(s: str, fmt: dict) -> str:
    return s.format(**fmt)


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img
    imgf = img.astype(np.float32)
    mn = float(np.nanmin(imgf))
    mx = float(np.nanmax(imgf))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros(imgf.shape, dtype=np.uint8)
    return (255.0 * (imgf - mn) / (mx - mn)).clip(0, 255).astype(np.uint8)


def load_video_from_cfg(cfg: dict, log: logging.Logger) -> np.ndarray:
    data = cfg["data"]
    path = data["path"]
    typ = data.get("type", "zarr")
    frames = int(data["frames"])
    H = int(data["height"])
    W = int(data["width"])

    t0 = time.perf_counter()
    if typ == "zarr":
        log.info("Loading zarr: %s", path)
        spadbinframes, _, _ = read_quanta_zarr(path)
    else:
        log.info("Loading bin: %s", path)
        spadbinframes = read_spadbin(
            path,
            nframes=frames,
            h=H,
            w=W,
            downsample=int(512 // W) if W <= 512 else 1,
            keep_prob=1.0,
            hotpix_thresh=3.0,
        )
    spadbinframes = np.asarray(spadbinframes)
    if spadbinframes.ndim != 3:
        raise ValueError(f"Expected (T,H,W) binframes. Got shape={spadbinframes.shape}")
    log.info("Loaded frames shape=%s in %.2fs", spadbinframes.shape, time.perf_counter() - t0)
    return spadbinframes


def apply_thinning(binframes: np.ndarray, T_exp: float, thin: int, log: logging.Logger) -> np.ndarray:
    if thin <= 1:
        return binframes

    t0 = time.perf_counter()
    log.info("Thinning: thin=%d (keep 1/%d events)", thin, thin)

    events = np.asarray(frames_to_events(binframes, T_exp))
    if events.ndim == 2 and events.shape[0] == 3:
        events = events.T  # (N,3)

    log.info("Events before thinning: %s", events.shape)
    events_th = thin_events_uniform(events, 1.0 / float(thin))
    log.info("Events after thinning:  %s", events_th.shape)

    t, y, x = events_th[:, 0], events_th[:, 1], events_th[:, 2]
    out = events_to_frames(x, y, t, binframes.shape[0], binframes.shape[1], binframes.shape[2])

    log.info("Thinning done -> binframes %s in %.2fs", out.shape, time.perf_counter() - t0)
    return out


# -------------------------
# main
# -------------------------
def main():
    args = parse_args()
    log = setup_logger(args.verbose, args.log_file)

    t_total0 = time.perf_counter()

    # ---- read cfg ----
    t0 = time.perf_counter()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    log.info("Loaded config in %.2fs: %s", time.perf_counter() - t0, args.config)

    data = cfg["data"]
    T_exp = float(data["T_exp"])
    slice_start, slice_end = map(int, data["slice"])

    win_cfg = cfg["window_settings"]
    temporal_size = int(win_cfg.get("temporal_size", 250))
    window_len = int(args.window) if args.window is not None else temporal_size
    if window_len <= 0:
        raise ValueError("window length must be > 0")

    skip = int(args.skip)
    if skip <= 0:
        raise ValueError("--skip must be > 0")

    thin = int(args.thin)
    if thin <= 0:
        raise ValueError("--thin must be >= 1")

    bounds = data["velocity_bounds"]
    vmin, vmax, n_vel = bounds

    fmt = {
        "slice_start": slice_start,
        "slice_end": slice_end,
        "width": int(data["width"]),
        "height": int(data["height"]),
        "vmin": vmin,
        "vmax": vmax,
        "n_vel": n_vel,
        "window_size": win_cfg.get("window_size", "NA"),
        "window_stride": win_cfg.get("window_stride", "NA"),
        "temporal_size": temporal_size,
        "patch_num": win_cfg.get("patch_num", 0),
    }

    base_dir = fmt_path(cfg["output"]["base_path"], fmt)
    sub_folder = fmt_path(cfg["output"]["sub_folder"], fmt)
    out_dir = Path(base_dir) / sub_folder
    ensure_dir(out_dir)

    vx_str = f"{args.vx:+.4f}".replace(".", "p")
    vy_str = f"{args.vy:+.4f}".replace(".", "p")
    out_name = (
        f"comp_vx_{vx_str}_vy_{vy_str}"
        f"_thin_{thin}_skip_{skip}_win_{window_len}"
        f"_slice_{slice_start}_{slice_end}.mp4"
    )
    out_video_path = out_dir / out_name

    log.warning("Output: %s", out_video_path)
    log.warning("Params: vx=%g vy=%g thin=%d skip=%d window=%d fps_out=%d",
                args.vx, args.vy, thin, skip, window_len, args.fps_out)

    # ---- load frames ----
    spadbinframes = load_video_from_cfg(cfg, log)

    if slice_end > spadbinframes.shape[0]:
        raise ValueError(f"slice_end={slice_end} exceeds available frames={spadbinframes.shape[0]}")

    t0 = time.perf_counter()
    frames = spadbinframes[slice_start:slice_end]
    log.info("Sliced frames [%d:%d] -> %s in %.2fs",
             slice_start, slice_end, frames.shape, time.perf_counter() - t0)

    eff_T = T_exp * (frames.shape[0] / float(data["frames"]))
    frames = apply_thinning(frames, eff_T, thin, log)

    dt = eff_T / frames.shape[0]
    log.info("Effective exposure eff_T=%.6fs, dt=%.6e s/frame", eff_T, dt)

    # ---- video writer ----
    H, W = frames.shape[1], frames.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, float(args.fps_out), (W, H), isColor=False)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_video_path}")

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda") if use_gpu else torch.device("cpu")
    log.warning("Device: %s (cuda=%s)", device, use_gpu)

    # ---- run compensation over windows ----
    start_frames = list(range(0, frames.shape[0] - window_len + 1, skip))
    n_out = len(start_frames)
    log.warning("Will generate %d stabilized frames (from %d input frames)", n_out, frames.shape[0])

    # timing accumulators
    t_fft = 0.0
    t_comp = 0.0
    t_write = 0.0

    pbar = tqdm(start_frames, desc="Compensating", unit="frame")
    for s in pbar:
        block = frames[s:s + window_len]

        # FFT
        t1 = time.perf_counter()
        C, ft, fy, fx = motion_detection.fft3_unshifted(block, dt)
        t_fft += time.perf_counter() - t1

        # compensate
        t1 = time.perf_counter()
        if use_gpu:
            C_t = torch.as_tensor(C, device=device, dtype=torch.complex64)
            ft_t = torch.as_tensor(ft, device=device, dtype=torch.float32)
            fy_t = torch.as_tensor(fy, device=device, dtype=torch.float32)
            fx_t = torch.as_tensor(fx, device=device, dtype=torch.float32)

            img_t = motion_detection.compensate_and_stabilize_tiled_torch(
                C_t=C_t, ft_t=ft_t, fy_t=fy_t, fx_t=fx_t,
                vx=float(args.vx), vy=float(args.vy),
                sign="minus",
                tile_y=128, tile_x=128,
            )
            img = img_t.detach().cpu().numpy()

            del C_t, ft_t, fy_t, fx_t, img_t
            torch.cuda.empty_cache()
        else:
            Cdes = motion_detection.compensate_motion(C, ft, fy, fx, args.vx, args.vy)
            img = motion_detection.stabilize_image_from_Fourier(Cdes, ft)
        t_comp += time.perf_counter() - t1

        # write
        t1 = time.perf_counter()
        writer.write(to_uint8(img))
        t_write += time.perf_counter() - t1

        # update bar stats
        done = (pbar.n + 1)
        if done > 0:
            pbar.set_postfix({
                "fft_ms": f"{1e3*t_fft/done:.1f}",
                "comp_ms": f"{1e3*t_comp/done:.1f}",
                "write_ms": f"{1e3*t_write/done:.1f}",
            })

    writer.release()

    # ---- summary timings ----
    total = time.perf_counter() - t_total0
    log.warning("Saved: %s", out_video_path)
    log.warning("Timing: total=%.2fs | FFT=%.2fs | COMP=%.2fs | WRITE=%.2fs",
                total, t_fft, t_comp, t_write)
    if n_out > 0:
        log.warning("Avg/frame: FFT=%.2fms COMP=%.2fms WRITE=%.2fms",
                    1e3*t_fft/n_out, 1e3*t_comp/n_out, 1e3*t_write/n_out)


if __name__ == "__main__":
    main()