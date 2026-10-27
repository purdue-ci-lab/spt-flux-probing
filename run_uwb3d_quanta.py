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
    parser.add_argument("--save-flux-pngs", action="store_true",
                        help="Also save a folder of flux PNGs (no gamma correction).")
    return parser.parse_args()


args = parse_args()
# without padding, 100kx512x512 takes ~25 GB
# with padding, like 102000x640x640 takes ~42 GB
use_cuda = args.cuda
load_frames_on_vram = args.load_frames_on_vram
load_output_on_vram = args.load_output_on_vram
save_flux_vid = args.save_flux_vid
save_gamma_pngs = args.save_gamma_pngs
save_flux_pngs = args.save_flux_pngs
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
# recon_slices.yaml indices are into the full data, so they are only valid when frame_slice keeps everything
frames_full = frame_slice is None or all(run_config.is_full_slice(s) for s in frame_slice)
recon_slice_run = run_config.parse_slice(run["recon_slice"])
window = run["window"]
window_kwargs = run["window_kwargs"]
window_settings = run["window_settings"]

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

    # a chunk covering the whole axis is the same as no chunking, so use None (full size, no windowing).
    # the hop is unused without chunking, so it is set to None as well
    chunk_t_vid, chunk_h_vid, chunk_w_vid = chunk_t, chunk_h, chunk_w
    hop_t_vid, hop_h_vid, hop_w_vid = hop_t, hop_h, hop_w
    if chunk_t_vid is not None and chunk_t_vid >= binvidT:
        logger.warning(f"chunk_t={chunk_t_vid} >= data length {binvidT} for {dataname}; setting chunk_t and hop_t to None")
        chunk_t_vid, hop_t_vid = None, None
    if chunk_h_vid is not None and chunk_h_vid >= binvidH:
        logger.warning(f"chunk_h={chunk_h_vid} >= data height {binvidH} for {dataname}; setting chunk_h and hop_h to None")
        chunk_h_vid, hop_h_vid = None, None
    if chunk_w_vid is not None and chunk_w_vid >= binvidW:
        logger.warning(f"chunk_w={chunk_w_vid} >= data width {binvidW} for {dataname}; setting chunk_w and hop_w to None")
        chunk_w_vid, hop_w_vid = None, None

    for keep_prob, target_fps in itertools.product(all_keepprobs, all_target_fps):
        alpha_vid = alpha
        Nt = int(nframes * target_fps / fps_vid)
        recon_shape = (Nt, Ny, Nx)
        if recon_slice_run is not None:
            recon_slice_vid = recon_slice_run
        elif (dataname, target_fps) in recon_slices and frames_full:
            recon_slice_vid = recon_slices[(dataname, target_fps)]
        else:
            if (dataname, target_fps) in recon_slices:
                logger.warning(
                    f"Skipping the recon_slices entry for ({dataname}, {target_fps}) since frame_slice "
                    f"does not keep the full data; reconstructing the full (sliced) data instead"
                )
            recon_slice_vid = None
        logger.info(f"Starting STFT for {dataname} with keep_prob={keep_prob} and target_fps={target_fps}")
        playfps = int(round(Nt / T_exp))
        alphastr = "" if alpha_vid is None else f"{alpha_vid:.5f}"
        if recon_slice_vid is None:
            slicestr = "slicefull"
        else:
            s0 = recon_slice_vid[0]
            slicestr = f"slice{run_config.name_part(s0.start)}_{run_config.name_part(s0.stop)}"
            if s0.step is not None and s0.step != 1:
                slicestr += f"_{s0.step}"
            slicestr += run_config.hw_slice_name(recon_slice_vid)

        # dataname, keep_prob and playfps are already in the parent dirs, so they are
        # left out here in an attempt to keep the full path short enough for Windows' 260-char limit
        recon_str_format = "-".join(part for part in [
            framestr, slicestr,
            f"chunks{run_config.name_part(chunk_t_vid)}x{run_config.name_part(chunk_h_vid)}x{run_config.name_part(chunk_w_vid)}",
            f"hops{run_config.name_part(hop_t_vid)}x{run_config.name_part(hop_h_vid)}x{run_config.name_part(hop_w_vid)}",
            f"bins{Nt}x{Ny}x{Nx}", f"window_{window}", f"alpha{alphastr}"
        ] if part)
        # the zip lands flat in ZIP_TRANSFER_DIR, so it keeps the full identifying name
        zip_run_id = "-".join([
            dataname, f"thin{keep_prob:.3f}", f"fps{playfps}", recon_str_format
        ])
        output_dir: Path = output_parent_dir / dataname / f"thin{keep_prob:.3f}" / f"fps{playfps}" / recon_str_format
        path_warning = run_config.long_path_warning(output_dir)
        if path_warning:
            logger.warning(path_warning)
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
            patch_sizes=(chunk_t_vid, chunk_h_vid, chunk_w_vid),
            hop_sizes=(hop_t_vid, hop_h_vid, hop_w_vid),
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
            # the settings the run dir name no longer spells out: the configs
            # verbatim plus the values resolved for this iteration
            "run_config": run,
            "data_type": data_type,
            "run_config_path": str(args.run),
            "recon_slices_path": str(args.recon_slices),
            "recon_shape": list(recon_shape),
            "window_kwargs": window_kwargs,
            "chunk_sizes": [chunk_t_vid, chunk_h_vid, chunk_w_vid],
            "hop_sizes": [hop_t_vid, hop_h_vid, hop_w_vid],
            "src_fps": fps_vid,
            "playback_fps": playfps,
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
        if save_flux_pngs:
            spvid.write_video(
                flux,
                output_dir / f"flux",
                gamma=1.0,
                cmap="grey", fileformat="png", vmin=0, vmax=maxflux
            )

        # zip results to prep for download
        if args.zip:
            zipoutpath = utils.zip_run(
                output_dir, ZIP_TRANSFER_DIR, run_id=zip_run_id, append_time=False
            )
            logger.info(f"Zipped results to {zipoutpath}")

        if keep_prob < 1.0:
            # thinned will be a copy
            del quantaframes
        del flux
