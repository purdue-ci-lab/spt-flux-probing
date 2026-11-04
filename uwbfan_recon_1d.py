from pathlib import Path
import time
import pickle

import numpy as np

from sptflux import ihpp_data, utils, logger, config
from spadlib.io import save_arrs_to_zarr
from spadlib.video_io import write_video

logger = logger.get_logger(__name__)


cuda = True
if cuda:
    import cupy as cp
    cp.cuda.runtime.setDevice(0)

T_exp = 0.1
src_W = 138
src_H = 91
# 200 GHz may need more than 1k frames to capture full transient
all_fps =     [300, 1_000, 2_000, 10_000, 100_000, 200_000, 30_000_000, 40_000_000, 200_000_000, 10_000_000_000, 30_000_000_000, 199_994_400_000]
all_nframes = [30,  100,   200,   1000,   1000,    1000,    1000,       1000,       1000,        1000,           1000,           1000]
all_tstart =  [0,   0,     0,     0,      0,       0,       0,          0,          0,           0,              0,              0.03000004160118163]
all_eps =     [1e-6, 1e-6, 1e-6,  1e-6,   1e-6,    1e-6,    1e-9,       1e-9,       1e-9,        1e-12,          1e-12,          1e-12]

output_parent = config.OUTPUT_DIR / Path("results/async-spad/AC_fan/spadfan-1duwb-thin1.000-freqharm-alphaNone")
with open(output_parent / "freqinfo-filtered.pkl", "rb") as f:
    freqinfo = pickle.load(f)
freqgrid_r = freqinfo["r"]["pix"]
ft_r = freqinfo["r"]["ft_full"]
freqgrid_g = freqinfo["g"]["pix"]
ft_g = freqinfo["g"]["ft_full"]
freqgrid_b = freqinfo["b"]["pix"]
ft_b = freqinfo["b"]["ft_full"]

if cuda:
    ft_r = cp.asarray(ft_r)
    ft_g = cp.asarray(ft_g)
    ft_b = cp.asarray(ft_b)

vid_settings = dict(
    playback_fps=60,
    vmin=0,
    gamma=1/2.2
)

def NUDFT(x, f, c, isign, cuda=False):
    xp = cp if cuda else np
    x = xp.asarray(x)
    f = xp.asarray(f)
    if hasattr(c, "__len__"):
        c = c[:, None]
        c = xp.asarray(c)
    fouriermat = c * xp.exp(isign * 2j * xp.pi * x[:, None] * f[None, :])
    F = xp.sum(fouriermat, axis=0)
    if cuda:
        F = F.get()
    return F

for fps, nframes, tstart, eps in zip(all_fps, all_nframes, all_tstart, all_eps):
    logger.info(f"Reconstructing UWB-Fan data at fps={fps} Hz, nframes={nframes}, tstart={tstart}s, eps={eps}")
    Nt = nframes
    datastr = "-".join([
        f"fps{utils.format_number_unit(fps)}",
        f"bins{Nt}",
    ])
    output_dir = output_parent / datastr
    output_dir.mkdir(parents=True, exist_ok=True)
    tend = tstart + Nt / fps
    logger.info(f"Reconstructing time from {tstart} to {tend} seconds")
    tt = np.linspace(tstart, tend, Nt) / T_exp
    if cuda:
        tt = cp.asarray(tt)

    nyquist = (fps / 2) / (1 / T_exp)

    # red
    flux_r = np.zeros((Nt, src_H, src_W))
    for i in range(src_H):
        for j in range(src_W):
            F = freqgrid_r[i][j]
            ft = ft_r
            mask = ft <= nyquist
            ft, F = ft[mask], F[mask]
            F[ft != 0] *= 2
            out = NUDFT(ft, tt, F, 1, cuda=cuda) / np.sqrt(T_exp)
            flux_r[:, i, j] = np.clip(out.real, 0, None)
    flux_r3ch = np.zeros((Nt, src_H, src_W, 3))
    flux_r3ch[:, :, :, 0] = flux_r
    write_video(flux_r3ch, output_dir / "gamma-r.avi", **vid_settings)
    del flux_r3ch

    # green
    flux_g = np.zeros((Nt, src_H, src_W))
    for i in range(src_H):
        for j in range(src_W):
            F = freqgrid_g[i][j]
            ft = ft_g
            mask = ft <= nyquist
            ft, F = ft[mask], F[mask]
            F[ft != 0] *= 2
            out = NUDFT(ft, tt, F, 1, cuda=cuda) / np.sqrt(T_exp)
            flux_g[:, i, j] = np.clip(out.real, 0, None)
    flux_g3ch = np.zeros((Nt, src_H, src_W, 3))
    flux_g3ch[:, :, :, 1] = flux_g
    write_video(flux_g3ch, output_dir / "gamma-g.avi", **vid_settings)
    del flux_g3ch

    # blue
    flux_b = np.zeros((Nt, src_H, src_W))
    for i in range(src_H):
        for j in range(src_W):
            F = freqgrid_b[i][j]
            ft = ft_b
            mask = ft <= nyquist
            ft, F = ft[mask], F[mask]
            F[ft != 0] *= 2
            out = NUDFT(ft, tt, F, 1, cuda=cuda) / np.sqrt(T_exp)
            flux_b[:, i, j] = np.clip(out.real, 0, None)
    flux_b3ch = np.zeros((Nt, src_H, src_W, 3))
    flux_b3ch[:, :, :, 2] = flux_b
    write_video(flux_b3ch, output_dir / "gamma-b.avi", **vid_settings)
    del flux_b3ch

    # combined RGB
    flux_rgb = np.zeros((Nt, src_H, src_W, 3), dtype=np.float64)
    # correct RGB response for filters in scene (refer to UWB supp)
    flux_rgb[:, :, :, 0] = flux_r * 1.0
    flux_rgb[:, :, :, 1] = flux_g * 0.75
    flux_rgb[:, :, :, 2] = flux_b * 0.4
    write_video(flux_rgb, output_dir / "gamma-rgb.avi", **vid_settings)
    del flux_rgb

    reconinfo = {
        "flux_r": np.asarray(flux_r, dtype="float32"),
        "flux_g": np.asarray(flux_g, dtype="float32"),
        "flux_b": np.asarray(flux_b, dtype="float32"),
    }
    save_arrs_to_zarr(
        reconinfo,
        output_dir / f"reconinfo.zarr",
        chunks={
            "flux_r": (400, src_H, src_W),
            "flux_g": (400, src_H, src_W),
            "flux_b": (400, src_H, src_W),
        },
        n_workers=48,
        overwrite=True,
    )
