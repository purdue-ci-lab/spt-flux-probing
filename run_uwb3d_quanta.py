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
        description="Windowed spatiotemporal flux probing with via STFT for quanta videos."
    )
    parser.add_argument("--run", required=True,
                        help="Path to a run-parameter YAML (config/runs/*.yaml).")
    parser.add_argument("--recon-slices", default=str(run_config.DEFAULT_RECON_SLICES),
                        help=f"Path to the per-dataset recon-slices YAML (default: {run_config.DEFAULT_RECON_SLICES}).")
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
# without padding, 100kx512x512 takes ~25 GB
# with padding, like 102000x640x640 takes ~42 GB
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
frame_slice = run_config.parse_slice(run["frame_slice"])
framestr = run_config.frame_slice_name(frame_slice)  # "" when nothing is sliced
recon_slice_run = run_config.parse_slice(run["recon_slice"])
window = run["window"]
window_kwargs = run["window_kwargs"]
window_settings = run["window_settings"]
max_frames_recon_default = run["max_frames_recon_default"]  # avoid accidental 50+ GB outputs

ZIP_TRANSFER_DIR = OUTPUT_DIR / "zipresults"

# spatiotemporal chunked NUFFT reconstruction using fixed chunk sizes and hop (overlap-add with windows)
# padding actually implemented so the whole video is reconstructed kind of properly
chunk_t, chunk_h, chunk_w = window_settings["chunk_t"], window_settings["chunk_h"], window_settings["chunk_w"]
hop_t, hop_h, hop_w = window_settings["hop_t"], window_settings["hop_h"], window_settings["hop_w"]

for dataname in all_datanames:
    quantapath = data_dir / f"{dataname}"
    # strip extension (handles .zarr, ignores dirs)
    dataname = utils.get_path_stem(quantapath)
    quantaframes_base, metadata = spio.read_quanta_auto(
        quantapath, load_data=True, H=src_H, W=src_W, return_meta=True
    )  # shape: (T, H, W)
    if metadata is not None:
        fps_vid = metadata.get("fps", src_fps)
    else:
        fps_vid = src_fps
    if frame_slice is not None:
        quantaframes_base = quantaframes_base[frame_slice]
        frameslice_tstep = frame_slice[0].step if frame_slice[0].step is not None else 1
        fps_vid = fps_vid // frameslice_tstep
    if hotpixel_correction:
        logger.info(f"Correcting hot pixels for {dataname}")
        quantaframes_base = spproc.correct_hotpixels(quantaframes_base)

    nframes = quantaframes_base.shape[0]
    Ny, Nx = quantaframes_base.shape[1], quantaframes_base.shape[2]
    binvidT, binvidH, binvidW = quantaframes_base.shape
    nevents_orig = da.count_nonzero(quantaframes_base).compute()
    T_exp = nframes / fps_vid
    V = T_exp * binvidH * binvidW

    for keep_prob, target_fps in itertools.product(all_keepprobs, all_target_fps):
        alpha_vid = alpha
        Nt = int(nframes * target_fps / fps_vid)
        recon_shape = (Nt, Ny, Nx)
        if recon_slice_run is not None:
            recon_slice_vid = recon_slice_run
        elif (dataname, target_fps) in recon_slices:
            recon_slice_vid = recon_slices[(dataname, target_fps)]
        elif Nt > max_frames_recon_default:
            recon_slice_vid = run_config.parse_slice(f"0:{max_frames_recon_default}")
        else:
            recon_slice_vid = None
        logger.info(f"Starting STFT for {dataname} with keep_prob={keep_prob} and target_fps={target_fps}")
        playfps = int(round(Nt / T_exp))
        alphastr = "None" if alpha_vid is None else f"{alpha_vid:.5f}"
        if recon_slice_vid is None:
            slicestr = "slicefull"
        else:
            s0 = recon_slice_vid[0]
            if s0.step is not None and s0.step != 1:
                slicestr = f"slice{s0.start}_{s0.stop}_{s0.step}"
            else:
                slicestr = f"slice{s0.start}_{s0.stop}"
            slicestr += run_config.hw_slice_name(recon_slice_vid)

        recon_str_format = "-".join(part for part in [
            f"{dataname}", framestr, slicestr, f"chunks{chunk_t}x{chunk_h}x{chunk_w}",
            f"hops{hop_t}x{hop_h}x{hop_w}", f"bins{Nt}x{Ny}x{Nx}", f"keep{keep_prob:.3f}", f"playfps{playfps}",
            f"window_{window}", f"alpha{alphastr}"
        ] if part)
        output_dir: Path = output_parent_dir / dataname / f"thin{keep_prob:.3f}" / f"fps{playfps}" / recon_str_format
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        if keep_prob < 1.0:
            dcr_prob = spproc.prob_from_dcr(25.0, fps_vid)
            t0_thin = time.time()
            quantaframes = spproc.thin_frames_uniform_dask(quantaframes_base, keep_prob, dcr_prob=dcr_prob, seed=42)
            t1_thin = time.time()
            logger.info(f"Thinning frames took {t1_thin - t0_thin:.2f} seconds")
            nevents_thinned = np.count_nonzero(quantaframes)
            logger.info(f"Thinned events from {nevents_orig} to {nevents_thinned} using keep_prob={keep_prob}")

            logger.info(f"Saving thinned frames for sanity to {output_dir / 'thinned_frames.zarr'}")
            spio.write_frames_zarr(
                output_dir / "thinned_frames.zarr",
                quantaframes, fps=fps_vid
            )
        else:
            # this variable should generally not be mutated
            quantaframes = quantaframes_base

        t0 = time.time()
        flux = flux_probing.uwb3d_stft_quanta(
            quantaframes,
            patch_sizes=(chunk_t, chunk_h, chunk_w),
            hop_sizes=(hop_t, hop_h, hop_w),
            window=window, window_kwargs=window_kwargs,
            recon_shape=recon_shape, recon_slice=recon_slice_vid,
            filter_func=None,
            V=V, alpha=alpha_vid, use_recon_nyquist=True,
            save_fourier_coeffs=False, all_fourier_coeffs=False,
            cuda=use_cuda, load_frames_on_vram=load_frames_on_vram, load_output_on_vram=load_output_on_vram
        )
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
        framenum_slice = framenum_all if recon_slice_vid is None else framenum_all[recon_slice_vid[0]]
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
            "window": window,
            "nframes_total": Nt,
            "nframes_recon": flux.shape[0],
            # (t, y, x) colon strings, as written in the run configs
            "frame_slice": run_config.slice_strs(frame_slice),
            "recon_slice": run_config.slice_strs(recon_slice_vid),
        }
        print("calculating flux statistics")
        print(f"flux dtype: {flux.dtype}, max value: {np.max(flux)}, min value: {np.min(flux)}, mean value: {np.mean(flux)}")
        print(f"maxflux (99th percentile): {np.quantile(flux, 0.99)}")
        maxflux = np.max(flux)

        # save results
        # utils.save_sparse_to_zarr(
        #     F, output_dir / f"fourier.zarr"
        # )
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

        # zip results to prep for download
        if args.zip:
            zipoutpath = utils.zip_run(
                output_dir, ZIP_TRANSFER_DIR, run_id=recon_str_format, append_time=False
            )
            logger.info(f"Zipped results to {zipoutpath}")

        if keep_prob < 1.0:
            # thinned will be a copy
            del quantaframes
        del flux
