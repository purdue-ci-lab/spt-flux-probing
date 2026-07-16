import itertools
from pathlib import Path
import time

import dask.array as da
import numpy as np
import zarr

from uwb3d import ihpp_fft, ihpp_data, utils, quanta_data
from uwb3d.logger import get_logger
from uwb3d.config import DATA_DIR, OUTPUT_DIR

logger = get_logger(__name__)


use_cuda = True
# without padding, 100kx512x512 takes ~25 GB
# with padding, like 102000x640x640 takes ~42 GB
load_frames_on_vram = True
load_output_on_vram = True
if use_cuda:
    import cupy as cp
    device = 1
    cp.cuda.runtime.setDevice(device)

MAX_FRAMES_RECON_DEFAULT = 2000  # to prevent high fps from accidentally taking 50+ GB, unless intended
ALPHA_UNTHINNED = 0.0001
ALPHA_THINNED = 0.0001
ZIP_TRANSFER_DIR = OUTPUT_DIR / "zipresults"

########## START DATA SETUP ##########
########## 100k fps quanta data ##########
# automatically determine output dimensions
# Nt = None
# all_target_fps = [500]
# Ny = None
# Nx = None
# frame_slice = (slice(None, None, 200), slice(None), slice(None))
# # recon_slice = (slice(0, 2000), slice(None), slice(None))
# recon_slice = None
# alpha = 1e-4
# data_dir = DATA_DIR / "quanta"
# output_parent_dir = OUTPUT_DIR / "results/quanta"
# all_datanames = ["gun-2s"]
# all_keepprobs = [1.0]
# # window_settings = quanta_data.WINDOW_SETTINGS_SPEED
# window_settings = {
#     "chunk_t": 128,
#     "chunk_h": 64,
#     "chunk_w": 64,
#     "hop_t": 64,
#     "hop_h": 32,
#     "hop_w": 32
# }
# src_W = 512
# src_H = 512
# fps = 100_000  # default source video fps

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

########## spike data ##########
# Nt = None
# all_target_fps = [10_000]
# Ny = None
# Nx = None
# frame_slice = (slice(None), slice(None), slice(None))
# recon_slice = None
# alpha = 0.0001
# data_dir = DATA_DIR / Path("spike")
# output_parent_dir = OUTPUT_DIR / Path("results/spike")
# all_datanames = ["viaduct-bridge"]
# all_keepprobs = [1.0]
# window_settings = {
#     "chunk_t": 500,
#     "chunk_h": 64,
#     "chunk_w": 64,
#     "hop_t": 125,
#     "hop_h": 16,
#     "hop_w": 16
# }
# fps = 40_000

########## 128k fps upscaled quanta data ##########
Nt = None  # all the videos have a different number of frames
all_target_fps = [1000]
Ny = None
Nx = None
frame_slice = (slice(None, None, 128), slice(None), slice(None))
# resample at original 1000 fps, but with upsampled reconstruction
recon_slice = (slice(None), slice(None), slice(None))
alpha = 1e-4
data_dir = DATA_DIR / "phantom-synth/upsampled"
output_parent_dir = OUTPUT_DIR / "results/phantom-synth"
all_datanames = [
    # "cerealmilk-fps1000_rife4_linear8_ppp0.1",
    # "coffeebeans-fps1000_rife4_linear8_ppp0.1",
    # "fireblast-fps1000_rife4_linear8_ppp0.1",
    # "grasswaterfall-fps1000_rife4_linear8_ppp0.1",
    # "shoewaterstep-fps1000_rife4_linear8_ppp0.1",
    # "strawberrymilk-fps1000_rife4_linear8_ppp0.1",
    # "waterbubbles-fps1000_rife4_linear8_ppp0.1",
    "wineglassfall-fps1000_rife4_linear8_ppp0.1",
    # "binary_ppp0.01_fps115393_T115393_H256_W512",
    # "binary_ppp0.05_fps115393_T115393_H256_W512",
]
window_settings = {
    "chunk_t": 128,
    "chunk_h": 64,
    "chunk_w": 64,
    "hop_t": 64,
    "hop_h": 32,
    "hop_w": 32,
}
all_keepprobs = [1.0]
src_W = 512
src_H = 256
fps = 128_000

########## 10 million fps phantom camera data ##########
# Nt = 256
# Ny = 250
# Nx = 400
# recon_slice = None
# cfarscale = 2.0
# data_dir = Path("data/phantom")
# output_parent_dir = Path("output/results/phantom")
# all_datanames = ["0.5cmx6cm-1_ppp0.1"]
# all_keepprobs = [1.0]
# fps = 10_000_000

########## 100k fps cavitation data ##########
# Nt = None  # all the videos have a different number of frames
# all_target_fps = [100_000]
# Ny = None
# Nx = None
# frame_slice = (slice(None), slice(None), slice(None))
# recon_slice = (slice(None), slice(None), slice(None))
# # alpha = "auto"
# alpha = 1e-3
# data_dir = DATA_DIR / "phantom-synth/upsampled"
# output_parent_dir = OUTPUT_DIR / "results/phantom-synth"
# all_datanames = [
#     # "cavitation1_rife0_linear1_ppp0.1",
#     # "cavitation1_rife2_linear1_ppp0.1",
#     # "cavitation1_rife4_linear8_ppp0.1",
#     "cavitation1_rife4_linear8_ppp1",
# ]
# window_settings = {
#     "chunk_t": 10000,
#     "chunk_h": 32,
#     "chunk_w": 32,
#     "hop_t": 2500,
#     "hop_h": 8,
#     "hop_w": 8,
# }
# all_keepprobs = [1.0]
# # determine fps from zarr metadata
# fps = None
########## END DATA SETUP ##########

########## Other variable setup ##########
# spatiotemporal chunked NUFFT reconstruction using fixed chunk sizes and hop (overlap-add with windows)
# padding actually implemented so the whole video is reconstructed kind of properly
window = "hann"
window_kwargs = None
# window = "gaussian"
# window_kwargs = {"std": (3000.0, 5.0, 5.0)}
########## End variable setup ##########

for dataname in all_datanames:
    # leave without extension to search for the format-agnostic reader in ihpp_data.read_quanta_auto
    quantapath = data_dir / f"{dataname}"
    quantaframes_base, metadata = ihpp_data.read_quanta_auto(
        quantapath, load_data=True, H=src_H, W=src_W
    )  # shape: (T, H, W)
    if metadata is not None:
        fps_vid = metadata.get("fps", fps)
    else:
        fps_vid = fps
    if frame_slice is not None:
        quantaframes_base = quantaframes_base[frame_slice]
        frameslice_tstep = frame_slice[0].step if frame_slice[0].step is not None else 1
        fps_vid = fps_vid // frameslice_tstep

    nframes = quantaframes_base.shape[0]
    Ny, Nx = quantaframes_base.shape[1], quantaframes_base.shape[2]
    binvidT, binvidH, binvidW = quantaframes_base.shape
    nevents_orig = da.count_nonzero(quantaframes_base).compute()
    T_exp = nframes / fps_vid
    V = T_exp * binvidH * binvidW

    for keep_prob, target_fps in itertools.product(all_keepprobs, all_target_fps):
        if alpha == "auto":
            if keep_prob < 1.0:
                alpha_vid = ALPHA_THINNED
            else:
                alpha_vid = ALPHA_UNTHINNED
        else:
            alpha_vid = alpha
        Nt = int(nframes * target_fps / fps_vid)
        recon_shape = (Nt, Ny, Nx)
        if recon_slice is not None:
            recon_slice_vid = recon_slice
        elif (dataname, target_fps) in quanta_data.data_configs:
            data_config = quanta_data.data_configs[(dataname, target_fps)]
            recon_slice_vid = data_config.get("recon_slice", recon_slice)
        else:
            if Nt > MAX_FRAMES_RECON_DEFAULT:
                recon_slice_vid = (slice(0, MAX_FRAMES_RECON_DEFAULT), slice(None), slice(None))
            else:
                recon_slice_vid = None
        logger.info(f"Starting STFT for {dataname} with keep_prob={keep_prob} and target_fps={target_fps}")
        playfps = int(round(Nt / T_exp))
        alphastr = "None" if alpha_vid is None else f"{alpha_vid:.5f}"
        if recon_slice is None:
            slicestr = "slicefull"
        else:
            if recon_slice[0].step is not None and recon_slice[0].step != 1:
                slicestr = f"slice{recon_slice[0].start}_{recon_slice[0].stop}_{recon_slice[0].step}"
            else:
                slicestr = f"slice{recon_slice[0].start}_{recon_slice[0].stop}"
        window_settings_data = None
        if window_settings == "auto":
            if keep_prob < 1.0:
                window_settings_data = quanta_data.WINDOW_SETTINGS_DARK
            else:
                window_settings_data = quanta_data.WINDOW_SETTINGS_BRIGHT
            logger.info(f"Automatically selected window settings: {window_settings_data}")
        else:
            window_settings_data = window_settings
        chunk_t, chunk_h, chunk_w = window_settings_data["chunk_t"], window_settings_data["chunk_h"], window_settings_data["chunk_w"]
        hop_t, hop_h, hop_w = window_settings_data["hop_t"], window_settings_data["hop_h"], window_settings_data["hop_w"]

        recon_str_format = "-".join([
            f"{dataname}", slicestr, f"chunks{chunk_t}x{chunk_h}x{chunk_w}",
            f"hops{hop_t}x{hop_h}x{hop_w}", f"bins{Nt}x{Ny}x{Nx}", f"keep{keep_prob:.3f}", f"playfps{playfps}",
            f"window_{window}", f"alpha{alphastr}"
        ])
        output_dir: Path = output_parent_dir / dataname / f"thin{keep_prob:.3f}" / f"fps{playfps}" / recon_str_format
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        if keep_prob < 1.0:
            dcr_prob = ihpp_data.prob_from_dcr(25.0, fps_vid)
            t0_thin = time.time()
            quantaframes = ihpp_data.thin_frames_uniform_da(quantaframes_base, keep_prob, dcr_prob=dcr_prob, seed=42)
            t1_thin = time.time()
            logger.info(f"Thinning frames took {t1_thin - t0_thin:.2f} seconds")
            nevents_thinned = np.count_nonzero(quantaframes)
            logger.info(f"Thinned events from {nevents_orig} to {nevents_thinned} using keep_prob={keep_prob}")

            logger.info(f"Saving thinned frames for sanity to {output_dir / 'thinned_frames.zarr'}")
            ihpp_data.write_quantaframes_zarr(
                output_dir / "thinned_frames.zarr",
                quantaframes, fps=fps_vid
            )
        else:
            # this variable should generally not be mutated
            quantaframes = quantaframes_base

        t0 = time.time()
        flux = ihpp_fft.uwb3d_patch_quanta(
            quantaframes,
            patch_sizes=(chunk_t, chunk_h, chunk_w),
            hop_sizes=(hop_t, hop_h, hop_w),
            window=window, window_kwargs=window_kwargs,
            recon_shape=recon_shape, recon_slice=recon_slice,
            filter_func=None,
            V=V, alpha=alpha_vid, use_recon_nyquist=True,
            save_fourier_coeffs=False, all_fourier_coeffs=False,
            cuda=use_cuda, load_frames_on_vram=load_frames_on_vram, load_output_on_vram=load_output_on_vram
        )
        # flux = ihpp_fft.uwb3d_patch_quanta_speed(
        #     quantaframes,
        #     patch_sizes=(chunk_t, chunk_h, chunk_w),
        #     hop_sizes=(hop_t, hop_h, hop_w),
        #     window=window, window_kwargs=window_kwargs,
        #     recon_shape=recon_shape,
        #     V=V, cfarscale=cfarscale
        # )
        t1 = time.time()
        logger.info(f"Completed patch-based UWB-3D NUFFT reconstruction in {t1 - t0:.2f} seconds")

        # for the sake of saving storage space. 1000x512x512 float64 is nearly 2 GBs already.
        # check if the flux goes above float16 max
        # we don't need that much precision
        # if np.max(flux) < np.finfo(np.float16).max:
        #     flux_to_save = np.asarray(flux, dtype="float16")
        # else:
        #     flux_to_save = np.asarray(flux, dtype="float32")
        flux_to_save = np.asarray(flux, dtype="float32")

        framenum_all = np.arange(0, Nt)
        framenum_slice = framenum_all if recon_slice is None else framenum_all[recon_slice[0]]
        reconinfo = {
            "flux": flux_to_save,
            "framenums": framenum_slice
        }
        metadata = {
            "dataname": dataname,
            "target_fps": target_fps,
            "keep_prob": keep_prob,
            "alpha": alpha_vid,
            "window": window,
            "nframes_total": Nt,
            "nframes_recon": flux.shape[0],
            "frame_start": recon_slice[0].start if recon_slice is not None else 0,
            "frame_end": recon_slice[0].stop if recon_slice is not None else Nt,
            "frame_step": recon_slice[0].step if recon_slice is not None else 1,
        }
        # if windowing in any dimension: edges of the dimensions will have artifacts
        # as a workaround, exclude these from max computation
        # if chunk_t is not None and chunk_t > 0:
        #     exclude_t = int(Nt * chunk_t / nframes)
        #     exclude_t_slice = slice(exclude_t, Nt - exclude_t)
        # else:
        #     exclude_t_slice = slice(None)
        # if chunk_h is not None and chunk_h > 0:
        #     exclude_y = int(Ny * chunk_h / binvidH)
        #     exclude_y_slice = slice(exclude_y, Ny - exclude_y)
        # else:
        #     exclude_y_slice = slice(None)
        # if chunk_w is not None and chunk_w > 0:
        #     exclude_x = int(Nx * chunk_w / binvidW)
        #     exclude_x_slice = slice(exclude_x, Nx - exclude_x)
        # else:
        #     exclude_x_slice = slice(None)
        # maxflux = np.max(flux[exclude_t_slice, exclude_y_slice, exclude_x_slice])
        print("calculating flux statistics")
        print(f"flux dtype: {flux.dtype}, max value: {np.max(flux)}, min value: {np.min(flux)}, mean value: {np.mean(flux)}")
        print(f"maxflux (99th percentile): {np.quantile(flux, 0.99)}")
        maxflux = np.max(flux)

        # save results
        # utils.save_sparse_to_zarr(
        #     F, output_dir / f"fourier.zarr"
        # )
        utils.save_arrs_to_zarr(
            reconinfo,
            output_dir / f"reconinfo.zarr",
            chunks={
                "flux": (400, Ny, Nx),
            },
            attrs=metadata,
            n_workers=48,
            overwrite=True,
        )
        ihpp_data.to_video(
            flux,
            output_dir / f"gamma.mp4",
            gamma=1/2.2,
            playback_fps=30, cmap="grey", vmin=0, vmax=maxflux
        )
        ihpp_data.to_video(
            flux,
            output_dir / f"gamma-viridis.mp4",
            gamma=1/2.2,
            playback_fps=30, cmap="viridis", vmin=0, vmax=maxflux
        )
        # ihpp_data.to_video(
        #     flux,
        #     output_dir / f"flux.mp4",
        #     gamma=1.0,
        #     playback_fps=30, cmap="grey", vmin=0, vmax=maxflux
        # )
        # takes too damn long to generate, just save pngs after the fact
        # ihpp_data.to_video(
        #     flux,
        #     output_dir / f"gamma",
        #     gamma=1/2.2,
        #     cmap="grey", fileformat="png", vmin=0, vmax=maxflux
        # )

        # zip results to prep for download

        zipoutpath = utils.zip_run(
            output_dir, ZIP_TRANSFER_DIR, run_id=recon_str_format, append_time=False
        )
        logger.info(f"Zipped results to {zipoutpath}")

        if keep_prob < 1.0:
            # thinned will be a copy
            del quantaframes
        del flux
