from pathlib import Path
import time

import cv2
import dask.array as da
import numpy as np

from uwb3d import ihpp_fft, get_logger
from spadlib import io as spio, processing as spproc, video_io as spvid

logger = get_logger(__name__)


########## START DATA SETUP ##########
########## 100k fps quanta data ##########
Nt = 1000
Ny = 64
Nx = Ny
cfarscale = 2.0
data_dir = Path("data/quanta")
output_parent_dir = Path("output/results/quanta")
# all_datanames = ["gun-2s", "guitar", "fan"]
all_datanames = ["lift_jump"]
# all_keepprobs = [1.0, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256]
all_keepprobs = [1.0]
fps = 100_000
X_res = Nx
Y_res = X_res

########## 1 million fps videos ##########
# Nt = 1000
# Ny = 512
# Nx = 512
# cfarscale = 2.0
# data_dir = Path("data/quanta")
# output_parent_dir = Path("output/results/quanta")
# # all_datanames = ["elevator", "fountain"]
# all_datanames = ["elevator"]
# # all_keepprobs = [1.0, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256]
# all_keepprobs = [1.0]
# fps = 1_000_000
# X_res = 512
# Y_res = X_res

########## bit2bit data ##########
# Nt = 2000
# Ny = 512
# Nx = 512
# cfarscale = 2.0
# data_dir = Path("data/bit2bit-data")
# output_parent_dir = Path("output/results/bit2bit")
# # all_datanames = ["Monkey_002us_100000fs_dim", "Resolution_target_drill_002us_100000fs"]
# all_datanames = ["Resolution_target_drill_002us_100000fs"]
# all_keepprobs = [1.0]
# fps = 100_000
# X_res = 512
# Y_res = X_res

########## spike data ##########
# Nt = 1000
# Ny = 250
# Nx = 400
# cfarscale = 2.0
# data_dir = Path("data/spike")
# output_parent_dir = Path("output/results/spike")
# # all_datanames = ["balloon", "car-100kmh", "forest", "railway", "rotation1", "rotation2", "rotation2x", "train-350kmh", "viaduct-bridge"]
# all_datanames = ["rotation2"]
# all_keepprobs = [1.0]
# fps = 20_000
# X_res = 400
# Y_res = 250
########## END DATA SETUP ##########

########## Other variable setup ##########
ft_nyquist_hz = fps / 2  # physical nyquist
ft_recon_nyquist = Nt / 2  # within normalized space
########## End variable setup ##########
tt = np.linspace(0, 1, Nt)
yy = np.linspace(0, 1, Ny)
xx = np.linspace(0, 1, Nx)


# spatiotemporal chunked NUFFT reconstruction using fixed chunk sizes and hop (overlap-add with Hann window)
# No padding is used; only chunks that fully fit inside the volume are processed.
# This cell is intended to replace the previous chunking logic with hop-based overlapping patches.

# parameters: adjust these to choose chunk dims and hop sizes
chunk_t = None    # number of temporal bins per chunk (in binned frames)
chunk_h = 32     # number of spatial rows per chunk (in binned frames)
chunk_w = 32     # number of spatial cols per chunk (in binned frames)

hop_t = None       # hop size in temporal bins
hop_h = 16       # hop size in spatial rows
hop_w = 16       # hop size in spatial cols

window = "hann"

# iterate datasets and keep-probs similar to previous cell
for dataname in all_datanames:
    quantapath = data_dir / f"{dataname}.zarr"
    quantaframes_base, _, _ = spio.read_quanta_zarr(quantapath, load_data=False)  # shape: (T_bins, H_bins, W_bins)
    quantaframes_base = np.asarray(quantaframes_base[:, ::8, ::8], dtype="uint8")
    # quantaframes_base = utils.resize_video(quantaframes_base, X_res, Y_res, interpolation=cv2.INTER_NEAREST)
    nframes = quantaframes_base.shape[0]
    T_exp = nframes / fps
    binvidT, binvidH, binvidW = quantaframes_base.shape
    V = T_exp * binvidH * binvidW
    nevents_orig = da.count_nonzero(quantaframes_base).compute()

    for keep_prob in all_keepprobs:
        logger.info(f"Starting hann window patch reconstruction for dataset {dataname} with keep_prob={keep_prob}")
        playfps = int(round(Nt / T_exp))
        forward_str_format = "-".join([
            f"{dataname}", f"keep{keep_prob:.3f}", f"chunks{chunk_t}x{chunk_h}x{chunk_w}",
            f"hops{hop_t}x{hop_h}x{hop_w}", f"res{Y_res}x{X_res}", f"window_{window}", f"cfar{cfarscale:.1f}"
        ])
        recon_str_format = "-".join([
            f"{dataname}", f"keep{keep_prob:.3f}", f"playfps{playfps}", f"chunks{chunk_t}x{chunk_h}x{chunk_w}",
            f"hops{hop_t}x{hop_h}x{hop_w}", f"res{Y_res}x{X_res}", f"bins{Nt}x{Ny}x{Nx}",
            f"window_{window}", f"cfar{cfarscale:.1f}"
        ])
        output_dir = output_parent_dir / dataname / f"thin{keep_prob:.3f}" / f"fps{playfps}" / recon_str_format
        output_dir.mkdir(parents=True, exist_ok=True)
        if keep_prob < 1.0:
            quantaframes = spproc.thin_frames_uniform(quantaframes_base, keep_prob, seed=42)
            nevents_thinned = np.count_nonzero(quantaframes)
            logger.info(f"Thinned events from {nevents_orig} to {nevents_thinned} using keep_prob={keep_prob}")
        else:
            quantaframes = quantaframes_base

        t0 = time.time()
        flux = ihpp_fft.uwb3d_patch_quanta_speed(
            quantaframes,
            patch_sizes=(chunk_t, chunk_h, chunk_w),
            hop_sizes=(hop_t, hop_h, hop_w),
            window=window,
            recon_shape=(Nt, Ny, Nx),
            V=V, cfarscale=cfarscale
        )
        t1 = time.time()
        logger.info(f"Completed patch-based UWB-3D NUFFT reconstruction in {t1 - t0:.2f} seconds")

        gamma = flux ** (1 / 2.2)

        reconinfo = {
            "flux": flux,
            "x": xx,
            "y": yy,
            "t": tt,
        }
        # beginning and end frames have edge artifacts due to hop-chunking
        # exclude these from max computation
        exclude = 2
        maxflux = np.max(flux[exclude:-exclude-1])
        maxgamma = np.max(gamma[exclude:-exclude-1])

        # save results
        # utils.save_arrs_to_zarr(
        #     reconinfo,
        #     output_dir / f"reconinfo.zarr",
        #     chunks={
        #         "flux": (400, Ny, Nx),
        #     },
        #     n_workers=48,
        #     overwrite=True,
        # )
        # ihpp_data.to_video(
        #     flux,
        #     output_dir / f"flux.avi",
        #     playback_fps=30, cmap="grey", maxv=maxflux
        # )
        # utils.avi_to_mov(output_dir / f"flux.avi")
        # ihpp_data.to_video(
        #     flux,
        #     output_dir / f"fluxframes",
        #     cmap="grey", format="png", maxv=maxflux
        # )
        spvid.write_video(
            gamma,
            output_dir / f"gamma.avi",
            playback_fps=30, cmap="grey", vmax=maxgamma
        )
        # utils.avi_to_mov(output_dir / f"gamma.avi")
        # ihpp_data.to_video(
        #     gamma,
        #     output_dir / f"gammaframes",
        #     cmap="grey", format="png", maxv=maxgamma
        # )
