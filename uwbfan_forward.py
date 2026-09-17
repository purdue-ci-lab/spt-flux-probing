from pathlib import Path
import time

import cv2
import numpy as np

from sptflux import flux_probing, datastructs, logger, config
from spadlib.io import read_async_spad_zarr
from spadlib.processing import thin_events_uniform

logger = logger.get_logger(__name__)


cuda = False
########## 3D FFT variable setup ##########
T_exp = 0.1  # exposure time (how much data to read from the start time)

# max frequencies to sweep
step = 1  # multiply by 10 for actual Hz step along time
fy_max = 45
fx_max = 69

# chunk sizes for processing
chunk_size = 100_000_000
ft_range_max = 10_000

# filepaths
output_parent = config.OUTPUT_DIR / "results/async-spad/AC_fan"
data_folder = config.DATA_DIR / "async-spad/AC_fan"

alpha = 0.00001
eps = 1e-12
keep_prob = 1/30
########## End variable setup ##########

fx = np.arange(-fx_max, fx_max + step, step)
fy = np.arange(-fy_max, fy_max + step, step)
fan_freq = 105.7
laser_freq = 1.9999440e7
freqs_ab_harmonics = np.arange(0, 15e9 + laser_freq, laser_freq).reshape(1,-1) + np.arange(-10, 11).reshape(-1,1) * fan_freq
freqs_ab_new = np.sort(np.unique((freqs_ab_harmonics.flatten())))
ft_harmonic = freqs_ab_new[freqs_ab_new >= 0] / (1 / T_exp)
brute_force_laser_max = np.max(ft_harmonic) / (1 / T_exp)
ft_r = ft_harmonic
# ft_r = np.arange(0, brute_force_laser_max + 1)
ft_gb = np.arange(0, 11) * fan_freq / (1 / T_exp)
# ft_gb = np.arange(0, fan_freq * 11) / (1 / T_exp)

########## Pixelwise SPAD data: uncomment if needed ##########
# data appears to have HxW = 91x138 pixels
src_W = 138
src_H = 91
V = src_W * src_H * T_exp

# no zarr?
# points_r, _ = ihpp_data.read_spadfolder(data_folder / "R", h=src_H, w=src_W, T_exp=T_exp, normalize=True)
# t_r, y_r, x_r = points_r[:, 0], points_r[:, 1], points_r[:, 2]
# points_g, _ = ihpp_data.read_spadfolder(data_folder / "G", h=src_H, w=src_W, T_exp=T_exp, normalize=True)
# t_g, y_g, x_g = points_g[:, 0], points_g[:, 1], points_g[:, 2]
# points_b, _ = ihpp_data.read_spadfolder(data_folder / "B", h=src_H, w=src_W, T_exp=T_exp, normalize=True)
# t_b, y_b, x_b = points_b[:, 0], points_b[:, 1], points_b[:, 2]

(t_r, y_r, x_r), pix_ts_r = read_async_spad_zarr(data_folder / "R.zarr", load_data=True)
(t_g, y_g, x_g), pix_ts_g = read_async_spad_zarr(data_folder / "G.zarr", load_data=True)
(t_b, y_b, x_b), pix_ts_b = read_async_spad_zarr(data_folder / "B.zarr", load_data=True)

if keep_prob < 1.0:
    print(f"original red count: {t_r.size}")
    thin_r_idxs = thin_events_uniform(t_r, keep_prob=keep_prob, return_idx=True, seed=42)
    thin_g_idxs = thin_events_uniform(t_g, keep_prob=keep_prob, return_idx=True, seed=42)
    thin_b_idxs = thin_events_uniform(t_b, keep_prob=keep_prob, return_idx=True, seed=42)
    t_r, y_r, x_r = t_r[thin_r_idxs], y_r[thin_r_idxs], x_r[thin_r_idxs]
    print(f"thinned red count: {t_r.size}")
    t_g, y_g, x_g = t_g[thin_g_idxs], y_g[thin_g_idxs], x_g[thin_g_idxs]
    t_b, y_b, x_b = t_b[thin_b_idxs], y_b[thin_b_idxs], x_b[thin_b_idxs]

    # ihpp_data.pixel_timeseries_to_npys(
    #     ihpp_data.thin_events_pixel_timeseries(pix_ts_r, keep_prob=keep_prob, seed=42), data_folder / f"R_thin{keep_prob:.3f}"
    # )
    # ihpp_data.pixel_timeseries_to_npys(
    #     ihpp_data.thin_events_pixel_timeseries(pix_ts_g, keep_prob=keep_prob, seed=42), data_folder / f"G_thin{keep_prob:.3f}"
    # )
    # ihpp_data.pixel_timeseries_to_npys(
    #     ihpp_data.thin_events_pixel_timeseries(pix_ts_b, keep_prob=keep_prob, seed=42), data_folder / f"B_thin{keep_prob:.3f}"
    # )

t_r, y_r, x_r = t_r / T_exp, y_r / src_H, x_r / src_W
t_g, y_g, x_g = t_g / T_exp, y_g / src_H, x_g / src_W
t_b, y_b, x_b = t_b / T_exp, y_b / src_H, x_b / src_W

data_str_format = "-".join(["spadfan", f"thin{keep_prob:.3f}", f"freqharmx{fy_max}x{fx_max}", f"alpha{alpha:.5f}"])
output_folder = output_parent / data_str_format
output_folder.mkdir(parents=True, exist_ok=True)

# actual 3d FFT
t0 = time.time()

# red
F_r, bound_r, (ft_r, fy_r, fx_r) = flux_probing.uwb3d_global_cont(
    x_r, y_r, t_r,
    fx.astype(np.float64), fy.astype(np.float64), ft_r.astype(np.float64),
    V=V, alpha=alpha,
    chunk_size=chunk_size, ft_range_max=ft_range_max, full_spectrum=False,
    cuda=cuda, eps=eps
)
freqinfo_r = {
    "F": F_r,
    "ft": ft_r,
    "fy": fy_r,
    "fx": fx_r,
}
np.savez_compressed(output_folder / f"freqinfo-r.npz", **freqinfo_r)

# green
F_g, bound_g, (ft_g, fy_g, fx_g) = flux_probing.uwb3d_global_cont(
    x_g, y_g, t_g,
    fx.astype(np.float64), fy.astype(np.float64), ft_gb.astype(np.float64),
    V=V, alpha=alpha,
    chunk_size=chunk_size, ft_range_max=ft_range_max, full_spectrum=False,
    cuda=cuda, eps=eps
)
freqinfo_g = {
    "F": F_g,
    "ft": ft_g,
    "fy": fy_g,
    "fx": fx_g,
}
np.savez_compressed(output_folder / f"freqinfo-g.npz", **freqinfo_g)

# blue
F_b, bound_b, (ft_b, fy_b, fx_b) = flux_probing.uwb3d_global_cont(
    x_b, y_b, t_b,
    fx.astype(np.float64), fy.astype(np.float64), ft_gb.astype(np.float64),
    V=V, alpha=alpha,
    chunk_size=chunk_size, ft_range_max=ft_range_max, full_spectrum=False,
    cuda=cuda, eps=eps
)
freqinfo_b = {
    "F": F_b,
    "ft": ft_b,
    "fy": fy_b,
    "fx": fx_b,
}
np.savez_compressed(output_folder / f"freqinfo-b.npz", **freqinfo_b)

t1 = time.time()
logger.info(f"3D NUFFT for 3 channels reconstruction time: {t1 - t0:.2f} seconds")
