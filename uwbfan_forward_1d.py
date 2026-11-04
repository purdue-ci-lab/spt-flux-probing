from pathlib import Path
import time

import cv2
import numpy as np
from tqdm import tqdm

from sptflux import ihpp_data, logger, config
from spadlib.io import read_async_spad_zarr
from spadlib.processing import thin_events_uniform

logger = logger.get_logger(__name__)


cuda = True
if cuda:
    import cupy as cp
    cp.cuda.runtime.setDevice(0)
########## 3D FFT variable setup ##########
T_exp = 0.1  # exposure time (how much data to read from the start time)

# filepaths
output_parent = config.OUTPUT_DIR / "results/async-spad/AC_fan"
data_folder = config.DATA_DIR / "async-spad/AC_fan"

alpha = None
keep_prob = 1.0
########## End variable setup ##########

fan_freq = 105.7
laser_freq = 1.9999440e7
freqs_ab_harmonics = np.arange(0, 15e9 + laser_freq, laser_freq).reshape(1,-1) + np.arange(-10, 11).reshape(-1,1) * fan_freq
freqs_ab_new = np.sort(np.unique((freqs_ab_harmonics.flatten())))
ft_harmonic = freqs_ab_new[freqs_ab_new >= 0] / (1 / T_exp)
# ft_gb = np.arange(0, 11) * fan_freq / (1 / T_exp)
# ft_gb = np.arange(0, fan_freq * 11 / (1 / T_exp))
ft_gb = ft_harmonic

########## Pixelwise SPAD data: uncomment if needed ##########
# data appears to have HxW = 91x138 pixels
src_W = 138
src_H = 91
V = T_exp

def process_pixtimes(pixtimes, texp, keep_prob, cuda=False):
    H = len(pixtimes)
    W = len(pixtimes[0])
    output = [[None for _ in range(W)] for _ in range(H)]
    for i in range(H):
        for j in range(W):
            output[i][j] = pixtimes[i][j] / texp
            if keep_prob < 1.0:
                output[i][j] = thin_events_uniform(output[i][j], keep_prob)
            if cuda:
                output[i][j] = cp.asarray(output[i][j], dtype="float64")
    return output

_, pixtimes_r = read_async_spad_zarr(data_folder / "R.zarr", load_data=True)
_, pixtimes_g = read_async_spad_zarr(data_folder / "G.zarr", load_data=True)
_, pixtimes_b = read_async_spad_zarr(data_folder / "B.zarr", load_data=True)
pixtimes_r = process_pixtimes(pixtimes_r, T_exp, keep_prob, cuda=cuda)
pixtimes_g = process_pixtimes(pixtimes_g, T_exp, keep_prob, cuda=cuda)
pixtimes_b = process_pixtimes(pixtimes_b, T_exp, keep_prob, cuda=cuda)

alphastr = f"alpha{alpha:.5f}" if alpha is not None else "None"
data_str_format = "-".join(["spadfan-1duwb", f"thin{keep_prob:.3f}", f"freqharm", f"alpha{alphastr}"])
output_folder = output_parent / data_str_format
output_folder.mkdir(parents=True, exist_ok=True)

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

import time
import pickle
import numpy as np
from tqdm import tqdm

# def _filter_freq_coeff(ft: np.ndarray, F: np.ndarray, bd: float):
#     mask = np.abs(F) >= bd
#     return ft[mask], F[mask]

t0 = time.time()

freqinfo = {
    "r": {"bd": np.zeros((src_H, src_W)), "pix": [[None]*src_W for _ in range(src_H)], "ft_full": np.asarray(ft_harmonic)},
    "g": {"bd": np.zeros((src_H, src_W)), "pix": [[None]*src_W for _ in range(src_H)], "ft_full": np.asarray(ft_gb)},
    "b": {"bd": np.zeros((src_H, src_W)), "pix": [[None]*src_W for _ in range(src_H)], "ft_full": np.asarray(ft_gb)},
}

pbar = tqdm(total=src_H * src_W)
if cuda:
    ft_harmonic = cp.asarray(ft_harmonic, dtype="float64")
    ft_gb = cp.asarray(ft_gb, dtype="float64")
for i in range(src_H):
    for j in range(src_W):
        ts_r = pixtimes_r[i][j]
        ts_g = pixtimes_g[i][j]
        ts_b = pixtimes_b[i][j]

        # R
        F_r = NUDFT(ts_r, ft_harmonic, 1.0, -1, cuda=cuda) / np.sqrt(T_exp)
        freqinfo["r"]["pix"][i][j] = F_r
        del ts_r
        pixtimes_r[i][j] = None

        # G
        F_g = NUDFT(ts_g, ft_gb, 1.0, -1, cuda=cuda) / np.sqrt(T_exp)
        freqinfo["g"]["pix"][i][j] = F_g
        del ts_g
        pixtimes_g[i][j] = None

        # B
        F_b = NUDFT(ts_b, ft_gb, 1.0, -1, cuda=cuda) / np.sqrt(T_exp)
        freqinfo["b"]["pix"][i][j] = F_b
        del ts_b
        pixtimes_b[i][j] = None

        pbar.update(1)
pbar.close()

out_path = output_folder / "freqinfo-filtered.pkl"
with open(out_path, "wb") as f:
    pickle.dump(freqinfo, f, protocol=pickle.HIGHEST_PROTOCOL)

t1 = time.time()
logger.info(f"Filtered NUDFT for 3 channels saved to pickle: {t1 - t0:.2f} seconds")
logger.info(f"Saved: {out_path}")
