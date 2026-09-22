from pathlib import Path
import time

import numpy as np

from sptflux import flux_probing, datastructs, logger, config
from spadlib.io import save_arrs_to_zarr
from spadlib.utils import format_number_unit
from spadlib.video_io import write_video

logger = logger.get_logger(__name__)


cuda = False
if cuda:
    import cupy as cp
    device = 1
    cp.cuda.runtime.setDevice(device)
T_exp = 0.1
# 200 GHz may need more than 1k frames to capture full transient
all_fps =     [300, 1_000, 2_000, 10_000, 100_000, 200_000, 30_000_000, 40_000_000, 200_000_000, 10_000_000_000, 30_000_000_000, 199_994_400_000]
all_nframes = [30,  100,   200,   1000,   1000,    1000,    1000,       1000,       1000,        1000,           1000,           1000]
all_tstart =  [0,   0,     0,     0,      0,       0,       0,          0,          0,           0,              0,              0.03000004160118163]
all_eps =     [1e-6, 1e-6, 1e-6,  1e-6,   1e-6,    1e-6,    1e-9,       1e-9,       1e-9,        1e-12,          1e-12,          1e-12]
# all_fps =     [100_000]
# all_nframes = [1000]
# all_tstart =  [0.03000004160118163]
# all_eps =     [1e-6]
# reconstruction grid sizes
Ny = 91
Nx = 138

# chunk sizes for processing
chunk_size = 100_000_000
ft_range_max = 10000

# reconstruction time range
xx = np.linspace(0, 1, Nx, endpoint=False)
yy = np.linspace(0, 1, Ny, endpoint=False)

output_parent = config.OUTPUT_DIR / Path("results/async-spad/AC_fan/tmp/spadfan-thin0.033-freqharmx45x69-alpha0.00001")
freqinfo_r = np.load(output_parent / "freqinfo-r.npz")
F_r = freqinfo_r["F"]
ft_r = freqinfo_r["ft"]
fy_r = freqinfo_r["fy"]
fx_r = freqinfo_r["fx"]
freqinfo_g = np.load(output_parent / "freqinfo-g.npz")
F_g = freqinfo_g["F"]
ft_g = freqinfo_g["ft"]
fy_g = freqinfo_g["fy"]
fx_g = freqinfo_g["fx"]
freqinfo_b = np.load(output_parent / "freqinfo-b.npz")
F_b = freqinfo_b["F"]
ft_b = freqinfo_b["ft"]
fy_b = freqinfo_b["fy"]
fx_b = freqinfo_b["fx"]

vid_settings = dict(
    res_scale=1.0,
    playback_fps=60,
    cmap="grey",
)

for fps, nframes, tstart, eps in zip(all_fps, all_nframes, all_tstart, all_eps):
    logger.info(f"Reconstructing UWB-Fan data at fps={fps} Hz, nframes={nframes}, tstart={tstart}s, eps={eps}")
    Nt = nframes
    datastr = "-".join([
        f"fps{format_number_unit(fps)}",
        f"bins{Nt}x{Ny}x{Nx}",
    ])
    output_dir = output_parent / datastr
    output_dir.mkdir(parents=True, exist_ok=True)
    tend = tstart + Nt / fps
    logger.info(f"Reconstructing time from {tstart} to {tend} seconds")
    tt = np.linspace(tstart, tend, Nt, endpoint=False) / T_exp
    coordgrid = datastructs.Lazy3DGrid(tt, yy, xx)
    nyquist = (fps / 2) / (1 / T_exp)

    # red
    mask = ft_r <= nyquist
    F_r_dbl = flux_probing.double_real_spectrum_3d(F_r, ft_r, nyquist=nyquist)
    out = flux_probing.spt_ihpp_inufft3_chunked(
        fx_r[mask], fy_r[mask], ft_r[mask], F_r_dbl[mask],
        coordgrid,
        chunk_size=chunk_size, ft_range_max=ft_range_max, cuda=cuda, eps=eps
    )
    flux_r = np.clip(out.real, 0, None)
    flux_r3ch = np.zeros((Nt, Ny, Nx, 3), dtype=np.float64)
    flux_r3ch[:, :, :, 0] = flux_r
    write_video(flux_r3ch, output_dir / "gamma-r.avi", gamma=1/2.2, **vid_settings)
    del flux_r3ch

    # green
    mask = ft_g <= nyquist
    F_g_dbl = flux_probing.double_real_spectrum_3d(F_g, ft_g, nyquist=nyquist)
    out = flux_probing.spt_ihpp_inufft3_chunked(
        fx_g[mask], fy_g[mask], ft_g[mask], F_g_dbl[mask],
        coordgrid,
        chunk_size=chunk_size, ft_range_max=ft_range_max, cuda=cuda, eps=eps
    )
    flux_g = np.clip(out.real, 0, None)
    flux_g3ch = np.zeros((Nt, Ny, Nx, 3), dtype=np.float64)
    flux_g3ch[:, :, :, 1] = flux_g
    write_video(flux_g3ch, output_dir / "gamma-g.avi", gamma=1/2.2, **vid_settings)
    del flux_g3ch

    # blue
    mask = ft_b <= nyquist
    F_b_dbl = flux_probing.double_real_spectrum_3d(F_b, ft_b, nyquist=nyquist)
    out = flux_probing.spt_ihpp_inufft3_chunked(
        fx_b[mask], fy_b[mask], ft_b[mask], F_b_dbl[mask],
        coordgrid,
        chunk_size=chunk_size, ft_range_max=ft_range_max, cuda=cuda, eps=eps
    )
    flux_b = np.clip(out.real, 0, None)
    flux_b3ch = np.zeros((Nt, Ny, Nx, 3), dtype=np.float64)
    flux_b3ch[:, :, :, 2] = flux_b
    write_video(flux_b3ch, output_dir / "gamma-b.avi", gamma=1/2.2, **vid_settings)
    del flux_b3ch

    # combined RGB
    flux_rgb = np.zeros((Nt, Ny, Nx, 3), dtype=np.float64)
    # correct RGB response for filters in scene (refer to UWB supp)
    flux_rgb[:, :, :, 0] = flux_r * 1.0
    flux_rgb[:, :, :, 1] = flux_g * 0.75
    flux_rgb[:, :, :, 2] = flux_b * 0.4
    write_video(flux_rgb, output_dir / "gamma-rgb.avi", gamma=1/2.2, **vid_settings)
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
            "flux_r": (400, Ny, Nx),
            "flux_g": (400, Ny, Nx),
            "flux_b": (400, Ny, Nx),
        },
        n_workers=48,
        overwrite=True,
    )
