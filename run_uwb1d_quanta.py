import argparse
import itertools
from pathlib import Path
import time

import dask.array as da
import numpy as np
import zarr

from sptflux import flux_probing, utils, run_config
from spadlib import io as spio, processing as spproc, video_io as spvid
from sptflux.logger import get_logger
from sptflux.config import DATA_DIR, OUTPUT_DIR

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-pixel flux probing for quanta videos."
    )
    parser.add_argument("--run", required=True,
                        help="Path to a run-parameter YAML (config/runs/*.yaml).")
    parser.add_argument("--recon-slices", default=str(run_config.DEFAULT_RECON_SLICES),
                        help="Path to the per-dataset recon-slices YAML.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--cuda", action="store_true",
                        help="Use CUDA if a GPU is available (default: CPU).")
    parser.add_argument("--no-vram-frames", dest="load_frames_on_vram", action="store_false",
                        help="Keep input frames in host memory instead of VRAM.")
    parser.add_argument("--no-vram-output", dest="load_output_on_vram", action="store_false",
                        help="Keep the output in host memory instead of VRAM.")
    parser.add_argument("--zip", action="store_true",
                        help="Zip each run's output dir into the transfer dir when done.")
    parser.add_argument("--save-flux-vid", action="store_true",
                        help="Also save a video of the flux (default: only save gamma-corrected video).")
    parser.add_argument("--save-gamma-pngs", action="store_true",
                        help="Also save a folder of gamma-corrected PNGs (default: only save video).")
    return parser.parse_args()


args = parse_args()
use_cuda = args.cuda
load_frames_on_vram = args.load_frames_on_vram
load_output_on_vram = args.load_output_on_vram
save_flux_vid = args.save_flux_vid
save_gamma_pngs = args.save_gamma_pngs
if use_cuda:
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() == 0:
            raise RuntimeError("no CUDA devices found")
        cp.cuda.runtime.setDevice(args.device)
    except Exception as e:
        logger.warning(f"CUDA requested but unavailable ({e}); falling back to CPU.")
        use_cuda = False

run = run_config.load_yaml(args.run)
data_type = run_config.load_data_type(run["data_type"])
recon_slices = run_config.load_recon_slices(args.recon_slices)

src_H = data_type["src_H"]
src_W = data_type["src_W"]
src_fps = data_type["src_fps"]
data_dir = DATA_DIR / data_type["data_subdir"]

output_parent_dir = OUTPUT_DIR / run["output_subdir"]
all_datanames = run["datanames"]
all_target_fps = run["target_fps"]
all_keepprobs = run["keep_probs"]
alpha = run["alpha"]
hotpixel_correction = run["hotpixel_correction"]
frame_slice = run_config.as_txy_slice(run_config.parse_slice(run["frame_slice"]))
recon_slice_run = run_config.parse_slice(run["recon_slice"])
max_frames_recon_default = run["max_frames_recon_default"]  # avoid accidental 50+ GB outputs

ZIP_TRANSFER_DIR = OUTPUT_DIR / "zipresults"

for dataname in all_datanames:
    quantapath = data_dir / f"{dataname}"
    quantaframes_base, metadata = spio.read_quanta_auto(quantapath, load_data=True, H=src_H, W=src_W)  # shape: (T, H, W)
    if frame_slice is not None:
        quantaframes_base = quantaframes_base[frame_slice]
    if hotpixel_correction:
        logger.info(f"Correcting hot pixels for {dataname}")
        quantaframes_base = spproc.correct_hotpixels(quantaframes_base)
    nframes = quantaframes_base.shape[0]
    Ny, Nx = quantaframes_base.shape[1], quantaframes_base.shape[2]
    binvidT, binvidH, binvidW = quantaframes_base.shape
    nevents_orig = da.count_nonzero(quantaframes_base).compute()
    if metadata is not None:
        fps_vid = metadata.get("fps", src_fps)
    else:
        fps_vid = src_fps
    T_exp = nframes / fps_vid

    for keep_prob, target_fps in itertools.product(all_keepprobs, all_target_fps):
        alpha_vid = alpha
        Nt = int(nframes * target_fps / fps_vid)
        if recon_slice_run is not None:
            recon_slice_vid = recon_slice_run
        elif (dataname, target_fps) in recon_slices:
            recon_slice_vid = recon_slices[(dataname, target_fps)]
        elif Nt > max_frames_recon_default:
            recon_slice_vid = slice(0, max_frames_recon_default)
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
        flux = flux_probing.uwb1d_quanta(
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
        framenum_slice = framenum_all if recon_slice_vid is None else framenum_all[recon_slice_vid]
        reconinfo = {
            "flux": flux_to_save,
            "framenums": framenum_slice
        }
        metadata = {
            "dataname": dataname,
            "target_fps": target_fps,
            "keep_prob": keep_prob,
            "alpha": alpha_vid,
            "hotpixel_correction": hotpixel_correction,
            "nframes_total": Nt,
            "nframes_recon": flux.shape[0],
            "frame_start": recon_slice_vid.start if recon_slice_vid is not None else 0,
            "frame_end": recon_slice_vid.stop if recon_slice_vid is not None else Nt,
            "frame_step": recon_slice_vid.step if recon_slice_vid is not None else 1,
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

        # fluxes are generally pointless to save because they're too dark to see anything useful
        if save_flux_vid:
            spvid.write_video(
                flux,
                output_dir / f"flux.mp4",
                gamma=1.0,
                playback_fps=30, cmap="grey", vmin=0, vmax=maxflux
            )
        # takes a while to generate pngs frames
        if save_gamma_pngs:
            spvid.write_video(
                flux,
                output_dir / f"gamma",
                gamma=1/2.2,
                cmap="grey", fileformat="png", vmin=0, vmax=maxflux
            )

        if args.zip:
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
