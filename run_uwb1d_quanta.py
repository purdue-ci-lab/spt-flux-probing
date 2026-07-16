import itertools
from pathlib import Path
import time

import dask.array as da
import numpy as np
import zarr

from uwb3d import ihpp_fft, utils, quanta_data
from spadlib import io as spio, processing as spproc, video_io as spvid
from uwb3d.logger import get_logger
from uwb3d.config import DATA_DIR, OUTPUT_DIR

logger = get_logger(__name__)


use_cuda = True
load_frames_on_vram = True
load_output_on_vram = True
if use_cuda:
    import cupy as cp
    cp.cuda.runtime.setDevice(0)

MAX_FRAMES_RECON_DEFAULT = 2000  # to prevent high fps from accidentally taking 50+ GB, unless intended
ZIP_TRANSFER_DIR = OUTPUT_DIR / "zipresults"

########## START DATA SETUP ##########
########## 100k fps quanta data ##########
# automatically determine output dimensions
# Nt = None
# all_target_fps = [1_000, 10_000, 100_000]
# frame_slice = (slice(None), slice(None), slice(None))
# recon_slice = None
# alpha = None
# data_dir = DATA_DIR / "quanta"
# output_parent_dir = OUTPUT_DIR / "results/uwb1d/quanta"
# all_datanames = ["Feb28_balloonBounce_acq0_2221ppps"]
# all_keepprobs = [1.0, 1/20, 1/50, 1/100]
# fps = 100_000  # default fps

########## 128k fps upscaled quanta data ##########
Nt = None  # all the videos have a different number of frames
all_target_fps = [128_000]
frame_slice = (slice(None), slice(None), slice(None))
# resample at original 1000 fps, but with upsampled reconstruction
recon_slice = slice(None, None, 128)
alpha = None
data_dir = DATA_DIR / "phantom-synth/upsampled"
output_parent_dir = OUTPUT_DIR / "results/uwb1d/phantom-synth"
all_datanames = [
    "cerealmilk-fps1000_rife4_linear8_ppp0.1",
    "coffeebeans-fps1000_rife4_linear8_ppp0.1",
    "fireblast-fps1000_rife4_linear8_ppp0.1",
    "grasswaterfall-fps1000_rife4_linear8_ppp0.1",
    "shoewaterstep-fps1000_rife4_linear8_ppp0.1",
    "strawberrymilk-fps1000_rife4_linear8_ppp0.1",
    "waterbubbles-fps1000_rife4_linear8_ppp0.1",
    "wineglassfall-fps1000_rife4_linear8_ppp0.1"
]
all_keepprobs = [1.0, 1/10, 1/50]
fps = 128_000
########## END DATA SETUP ##########

for dataname in all_datanames:
    # leave without extension to search for the format-agnostic reader in ihpp_data.read_quanta_auto
    quantapath = data_dir / f"{dataname}"
    quantaframes_base, metadata = spio.read_quanta_auto(quantapath)  # shape: (T, H, W)
    if frame_slice is not None:
        quantaframes_base = quantaframes_base[frame_slice]
    nframes = quantaframes_base.shape[0]
    Ny, Nx = quantaframes_base.shape[1], quantaframes_base.shape[2]
    binvidT, binvidH, binvidW = quantaframes_base.shape
    nevents_orig = da.count_nonzero(quantaframes_base).compute()
    if metadata is not None:
        fps_vid = metadata.get("fps", fps)
    else:
        fps_vid = fps
    T_exp = nframes / fps_vid

    for keep_prob, target_fps in itertools.product(all_keepprobs, all_target_fps):
        alpha_vid = alpha
        Nt = int(nframes * target_fps / fps_vid)
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
        logger.info(f"Starting 1D UWB for {dataname} with keep_prob={keep_prob} and target_fps={target_fps}")
        playfps = int(round(Nt / T_exp))
        alphastr = "None" if alpha_vid is None else f"{alpha_vid:.5f}"
        if recon_slice_vid is None:
            slicestr = "slicefull"
        else:
            if recon_slice_vid.step is not None and recon_slice_vid.step != 1:
                slicestr = f"slice{recon_slice_vid.start}_{recon_slice_vid.stop}_step{recon_slice_vid.step}"
            else:
                slicestr = f"slice{recon_slice_vid.start}_{recon_slice_vid.stop}"
        recon_str_format = "-".join([
            "uwb1d", f"{dataname}", slicestr,
            f"bins{Nt}x{Ny}x{Nx}", f"keep{keep_prob:.3f}", f"playfps{playfps}",
            f"alpha{alphastr}"
        ])
        output_dir: Path = output_parent_dir / dataname / f"thin{keep_prob:.3f}" / f"fps{playfps}" / recon_str_format
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        if keep_prob < 1.0:
            dcr_prob = spproc.prob_from_dcr(25.0, fps_vid)
            quantaframes = spproc.thin_frames_uniform(quantaframes_base, keep_prob, dcr_prob=dcr_prob, seed=42)
            nevents_thinned = np.count_nonzero(quantaframes)
            logger.info(f"Thinned events from {nevents_orig} to {nevents_thinned} using keep_prob={keep_prob}")
        else:
            quantaframes = quantaframes_base

        t0 = time.time()
        flux = ihpp_fft.uwb1d_quanta(
            quantaframes,
            Nt=Nt,
            recon_slice=recon_slice_vid,
            T_exp=T_exp, alpha=alpha_vid,
            cuda=use_cuda,
            load_frames_on_vram=load_frames_on_vram, load_output_on_vram=load_output_on_vram
        )
        t1 = time.time()
        logger.info(f"Completed 1D UWB reconstruction in {t1 - t0:.2f} seconds")

        # if np.max(flux) < np.finfo(np.float16).max:
        #     flux_to_save = np.asarray(flux, dtype="float16")
        # else:
        #     flux_to_save = np.asarray(flux, dtype="float32")
        flux_to_save = np.asarray(flux, dtype="float32")

        framenum_all = np.arange(0, Nt)
        framenum_slice = framenum_all if recon_slice is None else framenum_all[recon_slice]
        reconinfo = {
            "flux": flux_to_save,
            "framenums": framenum_slice
        }
        metadata = {
            "dataname": dataname,
            "target_fps": target_fps,
            "keep_prob": keep_prob,
            "alpha": alpha_vid,
            "nframes_total": Nt,
            "nframes_recon": flux.shape[0],
            "frame_start": recon_slice.start if recon_slice is not None else 0,
            "frame_end": recon_slice.stop if recon_slice is not None else Nt,
            "frame_step": recon_slice.step if recon_slice is not None else 1,
        }
        print("calculating flux statistics")
        print(f"flux dtype: {flux.dtype}, max value: {np.max(flux)}, min value: {np.min(flux)}, mean value: {np.mean(flux)}")
        print(f"maxflux (99th percentile): {np.quantile(flux, 0.99)}")
        print(f"actual max flux: {np.max(flux)}")
        maxflux = np.max(flux)

        # save results
        spio.save_arrs_to_zarr(
            reconinfo,
            output_dir / f"reconinfo.zarr",
            chunks={
                "flux": (400, Ny, Nx),
            },
            attrs=metadata,
            n_workers=48,
            overwrite=True,
        )
        spvid.write_video(
            flux,
            output_dir / f"gamma.mp4",
            gamma=1/2.2,
            playback_fps=30, cmap="grey", vmin=0, vmax=maxflux
        )

        zipoutpath = utils.zip_run(
            output_dir, ZIP_TRANSFER_DIR, run_id=recon_str_format, append_time=False
        )
        logger.info(f"Zipped results to {zipoutpath}")

        del flux, flux_to_save, reconinfo
        if keep_prob < 1.0:
            del quantaframes

        # If using cupy, explicitly free GPU memory pool too
        if use_cuda:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    del quantaframes_base
