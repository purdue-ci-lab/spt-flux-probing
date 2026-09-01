"""
Worked example: spatiotemporal flux probing on a synthetic asynchronous SPAD stream.

An asynchronous SPAD does not produce frames. It produces a *stream* of photon
detections, each one a (t, y, x) triplet. Flux probing treats that stream as a
sample from an inhomogeneous Poisson process (IHPP) whose intensity is the scene
flux we want to recover, and estimates that intensity directly in the Fourier
domain, with no binning into frames anywhere in the pipeline.

The pipeline has two halves, mirroring `uwbfan_forward.py` and `uwbfan_recon.py`:

  1. FORWARD (`flux_probing.uwb3d_global_cont`)
     Sweep a grid of candidate 3D frequencies (fx, fy, ft), evaluate the forward
     type-3 NUFFT of the photon stream at each one, and keep only the
     coefficients whose magnitude exceeds a CFAR (constant false alarm rate)
     threshold. Shot noise from a Poisson process is flat in frequency, so the
     coefficients that stick out above that threshold are real scene content.
     The output is a short, sparse list of frequency triplets + coefficients.

  2. RECONSTRUCTION (`flux_probing.spt_ihpp_inufft3_chunked`)
     Evaluate the inverse type-3 NUFFT of that sparse spectrum on whatever
     spatiotemporal grid you want. Nothing about the forward pass assumed a
     framerate, so the *same* saved spectrum can be replayed at 50 fps, at
     1000 fps, or over any sub-window of the exposure, just by changing the
     query grid. That reuse is the main thing this script demonstrates.

Everything here lives in a unit box: photon coordinates are normalized to
[0, 1]^3 (x by the sensor width, y by the height, t by the exposure time). That
normalization carries over to the frequencies, which are therefore *not* in Hz
and cycles/pixel but in cycles per exposure and cycles per sensor width/height:

    ft_normalized = f_Hz * T_exp = f_Hz / (1 / T_exp)
    fx_normalized = f_cycles_per_px * width_px
    fy_normalized = f_cycles_per_px * height_px

so ft = 1 is one cycle across the whole exposure, i.e. 1 / T_exp Hz. This example
picks T_exp = 1 s precisely so the two coincide and the numbers stay readable: a
sweep value of 15 is 15 Hz. At any other exposure they do not, and the conversion
above is mandatory - see the note at the frequency sweep below.

Run with:

    uv run example_asyncspad.py
"""
import time

import matplotlib.pyplot as plt
import numpy as np

from sptflux import flux_probing, datastructs, logger, config
from spadlib.video_io import write_video

logger = logger.get_logger(__name__)


########## Variable setup ##########
cuda = False
seed = 0

# Synthetic capture. Coordinates are normalized, so the process lives in the
# unit box and its volume V is 1. V only sets the units of the recovered flux:
# with V = 1 the reconstruction is in photons per unit box volume, i.e. photons
# per (normalized area x exposure). Real captures pass the physical volume
# instead (e.g. V = width_px * height_px * T_exp in `uwbfan_forward.py`) so that
# the flux comes out in photons/pixel/second.
T_exp = 1.0
V = 1.0
n_photons = 2_000_000  # expected number of detections over the whole exposure

# Ground truth scene: a constant background plus a few traveling sinusoids.
# Writing each component as cos(2*pi*(fx*x + fy*y + ft*t)) puts its Fourier peak
# exactly on the triplet (fx, fy, ft) of the non-negative-ft half spectrum, which
# makes the CFAR output easy to check by eye further down.
# `amp` is a modulation depth relative to the background; they must sum to < 1 so
# the flux stays non-negative (a Poisson intensity cannot go below zero).
scene_components = [
    # slow drifting grating: survives even a low framerate
    dict(name="slow drift", fx=3.0, fy=2.0, ft=15.0, amp=0.45),
    # fast flicker: only representable once the framerate's Nyquist limit exceeds 200 Hz
    dict(name="fast flicker", fx=-5.0, fy=0.0, ft=200.0, amp=0.35),
]

# Frequency grid to sweep in the forward pass. Only ft >= 0 is swept: the flux is
# real, so the negative-ft half of the spectrum is the complex conjugate of this
# half and carries no extra information. `uwb3d_global_cont` detects that all
# ft >= 0 and accounts for the dropped half itself.
#
# NORMALIZED, NOT PHYSICAL, FREQUENCIES. Because the coordinates were squashed
# into the unit box, so were the frequencies. A physical frequency has to be
# converted before it goes into a sweep:
#
#     ft_normalized = f_Hz * T_exp = f_Hz / (1 / T_exp)
#     fx_normalized = f_cycles_per_px * width_px    (same for fy with height_px)
#
# and converted back on the way out (f_Hz = ft_normalized / T_exp) whenever a
# detected frequency is reported. `uwbfan_forward.py` does exactly this: it
# builds the laser and fan harmonics in Hz, then divides by 1 / T_exp to get the
# `ft` it actually sweeps. Here T_exp = 1 s, so the two happen to be numerically
# equal and 15 is both 15 Hz and 15 cycles per exposure - do not read that as a
# general rule.
#
# A step of 1 in ft is one cycle per exposure, i.e. the natural bin spacing of a
# DFT over this exposure, but nothing requires the sweep to sit on integers: this
# is a type-3 NUFFT, so arbitrary real frequencies are fine (again see
# `uwbfan_forward.py`, whose harmonics land wherever the physics puts them).
fx_sweep = np.arange(-8, 9, 1, dtype=np.float64)   # cycles per sensor width
fy_sweep = np.arange(-8, 9, 1, dtype=np.float64)   # cycles per sensor height
ft_sweep = np.arange(0, 301, 1, dtype=np.float64)  # cycles per exposure (= Hz only because T_exp = 1)

# CFAR false alarm rate. Roughly alpha * n_freqs noise-only frequencies are
# expected to sneak past the threshold, so with ~87k frequencies swept here,
# 1e-5 lets through about one spurious frequency on average.
alpha = 1e-5

# Performance knobs, both no-ops at this problem size but kept to show where they
# go. `chunk_size` caps how many points/frequencies are handed to FINUFFT at once
# (memory), and `ft_range_max` caps the *span* of ft within one chunk, which
# keeps FINUFFT's internal type-3 grid from blowing up when the temporal
# frequencies reach into the GHz (as they do for the real UWB fan data).
chunk_size = 5_000_000
ft_range_max = 10_000
eps = 1e-9  # NUFFT tolerance

# Reconstruction grid. The spatial grid is arbitrary (it is a resampling of a
# continuous field, not a sensor readout), the temporal grid is what defines the
# "framerate" of the output video.
Ny = 64
Nx = 64
# (fps, number of frames, start time in seconds). Nyquist for each is fps / 2:
#   50 fps   -> 25 Hz : only the 15 Hz drift is representable
#   200 fps  -> 100 Hz: still only the drift
#   1000 fps -> 500 Hz: the 200 Hz flicker comes back too
# The last one only replays a 50 ms slice of the exposure, showing that the
# reconstruction window is free as well.
recon_settings = [
    (50, 50, 0.0),
    (200, 50, 0.0),
    (1000, 50, 0.0),
]

# Video settings. `playback_fps` is purely a viewing choice and has nothing to do
# with the `fps` of a reconstruction: that one is the rate the flux is *sampled*
# at, this one is how fast the frames are played back on screen.
vid_settings = dict(
    res_scale=4.0,  # 64x64 frames are tiny, upscale (nearest neighbor) for viewing
    playback_fps=20,
    cmap="inferno",
)

output_folder = config.OUTPUT_DIR / "results/example-asyncspad"
########## End variable setup ##########

output_folder.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(seed)


def scene_flux(t, y, x, components=scene_components, nyquist=None):
    """
    Ground truth flux (photons per unit volume) of the synthetic scene.

    Args:
        t, y, x: coordinate arrays, broadcastable against each other, in [0, 1].
        components: list of component dicts (see `scene_components`).
        nyquist: if given, drop components whose temporal frequency is above this
            limit. Used to build the band-limited ground truth that a given
            framerate could possibly represent.

    Returns:
        Flux array of the broadcast shape of (t, y, x).
    """
    flux = np.ones(np.broadcast(t, y, x).shape, dtype=np.float64)
    for comp in components:
        if nyquist is not None and comp["ft"] > nyquist:
            continue
        flux = flux + comp["amp"] * np.cos(
            2 * np.pi * (comp["fx"] * x + comp["fy"] * y + comp["ft"] * t)
        )
    # n_photons / V is the mean flux, since the components all average to zero
    return (n_photons / V) * flux


def sample_photon_stream():
    """
    Draw photon detections from the IHPP defined by `scene_flux` via thinning.

    Thinning is exact (no binning, no approximation): draw a homogeneous Poisson
    process at the maximum possible flux over the unit box, then keep each
    candidate detection with probability flux(p) / flux_max. What survives is an
    exact sample of the inhomogeneous process, which is what a photon-starved
    asynchronous SPAD hands you.

    Returns:
        (t, y, x): 1D float arrays of detection coordinates in [0, 1].
    """
    flux_max = (n_photons / V) * (1 + sum(c["amp"] for c in scene_components))
    n_proposals = rng.poisson(flux_max * V)
    t = rng.random(n_proposals)
    y = rng.random(n_proposals)
    x = rng.random(n_proposals)
    keep = rng.random(n_proposals) < scene_flux(t, y, x) / flux_max
    # sorting by time is not required by the math, it just matches how a real
    # sensor streams its detections out
    order = np.argsort(t[keep])
    return t[keep][order], y[keep][order], x[keep][order]


def bin_photon_stream(tt, dt):
    """
    Naive framing of the photon stream: a 3D histogram of the detections over the
    same (t, y, x) grid the flux is reconstructed on.

    This is the baseline that flux probing is replacing, and it is worth watching
    next to the reconstruction: the bins here hold only a handful of photons each
    (well under one photon per bin at 1000 fps), so the scene is buried in shot
    noise, while the reconstruction from the exact same detections is clean.

    Args:
        tt: 1D array of frame start times, normalized by T_exp.
        dt: frame duration, normalized by T_exp.

    Returns:
        Photon counts of shape (len(tt), Ny, Nx).
    """
    # the grid coordinates are the lower corner of each voxel, so the bin edges
    # run from the first coordinate to one voxel past the last
    t_edges = np.append(tt, tt[-1] + dt)
    y_edges = np.linspace(0, 1, Ny + 1)
    x_edges = np.linspace(0, 1, Nx + 1)
    # detections outside the reconstruction window fall outside the edges and are
    # dropped, which is what we want
    counts, _ = np.histogramdd((t_ev, y_ev, x_ev), bins=(t_edges, y_edges, x_edges))
    return counts


########## Synthetic asynchronous SPAD stream ##########
# TO RUN THIS ON YOUR OWN DATA: replace this block. Everything
# downstream needs nothing but three 1D arrays of photon coordinates, e.g.
#
#     from spadlib.io import read_async_spad_zarr
#     (t_ev, y_ev, x_ev), _, _ = read_async_spad_zarr(data_folder / "R.zarr", load_data=True)
#     t_ev, y_ev, x_ev = t_ev / T_exp, y_ev / src_H, x_ev / src_W
#
# as in `uwbfan_forward.py`. THE COORDINATES MUST BE NORMALIZED TO [0, 1]: divide
# t by the exposure time and y, x by the sensor height and width.
# `uwb3d_global_cont` raises a ValueError if anything falls outside the unit box.
# Normalizing the coordinates is also what sets the units of the frequency sweep
# (see the note there), so update `fx_sweep`, `fy_sweep`, `ft_sweep` and V to
# match your capture; V is then the physical volume width_px * height_px * T_exp.
t_ev, y_ev, x_ev = sample_photon_stream()
logger.info(f"Sampled {t_ev.size} photon detections over T_exp = {T_exp} s")


########## Forward pass: photon stream -> sparse Fourier spectrum ##########
# This is the `uwbfan_forward.py` half. Note there is no framerate anywhere in
# this call: the input is a raw list of detection coordinates.
t0 = time.time()
F, bound, (ft, fy, fx) = flux_probing.uwb3d_global_cont(
    x_ev, y_ev, t_ev,
    fx_sweep, fy_sweep, ft_sweep,
    V=V, alpha=alpha,
    chunk_size=chunk_size, ft_range_max=ft_range_max,
    # full_spectrum=False throws away everything below the CFAR bound as it goes,
    # so the full |fx| x |fy| x |ft| energy grid is never held in memory. That is
    # mandatory for real sweeps, which can reach billions of frequencies.
    full_spectrum=False,
    cuda=cuda, eps=eps,
)
logger.info(f"Forward NUFFT took {time.time() - t0:.2f} s")

# `spt_ihpp_inufft3_chunked` chunks along ft and assumes ft is sorted ascending.
# It already is here (ft is the slowest-varying axis of the swept grid), but a
# spectrum assembled from several sweeps might not be, so sort defensively.
order = np.argsort(ft)
F, ft, fy, fx = F[order], ft[order], fy[order], fx[order]

# Save the spectrum on its own. It is tiny (a few detected frequencies vs. the
# millions of input photons), and it is the only thing the reconstruction needs,
# which is exactly why any number of framerates can be replayed from it later
# without ever touching the photon stream again.
np.savez_compressed(
    output_folder / "freqinfo.npz", F=F, ft=ft, fy=fy, fx=fx, bound=bound
)

logger.info(f"{F.size} frequencies passed the CFAR bound of {bound:.1f}")
logger.info("strongest detected frequencies (fx, fy, ft) | |F|:")
for i in np.argsort(np.abs(F))[::-1][:8]:
    logger.info(f"  ({fx[i]:6.1f}, {fy[i]:6.1f}, {ft[i]:6.1f}) | {np.abs(F[i]):12.1f}")
# Expected: the DC term at (0, 0, 0) with |F| = n_photons / sqrt(V), then one
# peak per scene component at its own (fx, fy, ft) with
# |F| = amp * n_photons / (2 * sqrt(V)).
logger.info("expected from ground truth:")
logger.info(f"  (   0.0,    0.0,    0.0) | {n_photons / np.sqrt(V):12.1f}  (DC)")
for comp in scene_components:
    logger.info(
        f"  ({comp['fx']:6.1f}, {comp['fy']:6.1f}, {comp['ft']:6.1f}) | "
        f"{comp['amp'] * n_photons / (2 * np.sqrt(V)):12.1f}  ({comp['name']})"
    )


########## Reconstruction: one spectrum, many framerates ##########
# This is the `uwbfan_recon.py` half. Everything below reads only `freqinfo.npz`,
# so in practice it would be a separate script run many times over.
freqinfo = np.load(output_folder / "freqinfo.npz")
F = freqinfo["F"]
ft = freqinfo["ft"]
fy = freqinfo["fy"]
fx = freqinfo["fx"]

xx = np.linspace(0, 1, Nx, endpoint=False)
yy = np.linspace(0, 1, Ny, endpoint=False)

for fps, n_frames, t_start in recon_settings:
    t_end = t_start + n_frames / fps
    # Query coordinates must be in the same normalized units as the photon
    # coordinates were, hence the division by T_exp.
    tt = np.linspace(t_start, t_end, n_frames, endpoint=False) / T_exp
    # Lazy3DGrid is the (t, y, x) meshgrid, materialized chunk by chunk so a
    # long, high-resolution reconstruction never has to fit in memory at once.
    coordgrid = datastructs.Lazy3DGrid(tt, yy, xx)

    # Anything above this framerate's Nyquist limit cannot be represented on the
    # output grid and would simply alias down into a fake slow oscillation, so it
    # gets dropped. This mask is the *only* thing that differs between framerates.
    # fps / 2 is in Hz, so it needs the same conversion into cycles per exposure
    nyquist = (fps / 2) / (1 / T_exp)
    keep = ft <= nyquist
    # The forward pass only swept ft >= 0, so each detected coefficient stands in
    # for itself plus its conjugate twin at -ft. Doubling the ft > 0 coefficients
    # (DC and exactly-Nyquist terms are their own twin, so they are left alone)
    # recovers the real-valued flux from the half spectrum. Note this returns a
    # copy: never overwrite `F` here, or the next framerate in the loop would
    # start from an already-doubled spectrum.
    F_recon = flux_probing.double_real_spectrum_3d(F, ft, nyquist=nyquist)

    logger.info(
        f"Reconstructing {n_frames} frames at {fps} fps from t = {t_start:.3f}s to "
        f"{t_end:.3f}s ({keep.sum()} of {F.size} frequencies below the {nyquist:.1f} Hz Nyquist limit)"
    )
    out = flux_probing.spt_ihpp_inufft3_chunked(
        fx[keep], fy[keep], ft[keep], F_recon[keep],
        coordgrid,
        chunk_size=chunk_size, ft_range_max=ft_range_max,
        V=V, cuda=cuda, eps=eps,
    )
    # The imaginary part is numerical noise once the half spectrum is doubled,
    # and a flux cannot be negative, so clip.
    flux_recon = np.clip(out.real, 0, None)

    # Ground truth for comparison, band-limited to what this framerate can
    # represent (at 50 and 200 fps, the 200 Hz flicker is simply not there).
    tg, yg, xg = np.meshgrid(tt, yy, xx, indexing="ij")
    flux_gt = scene_flux(tg, yg, xg, nyquist=nyquist)
    rel_rmse = np.sqrt(np.mean((flux_recon - flux_gt) ** 2)) / np.mean(flux_gt)
    logger.info(f"  relative RMSE vs band-limited ground truth: {rel_rmse:.4f}")
    # display scale shared by the videos and the figure below
    vmin, vmax = flux_gt.min(), flux_gt.max()

    np.save(output_folder / f"flux-{fps}fps.npy", flux_recon.astype(np.float32))

    # Videos, three per framerate: the band-limited ground truth, the raw photon
    # stream binned onto the same grid, and the reconstruction. Ground truth and
    # reconstruction share a display scale so they can be compared frame for
    # frame; the photon counts are in different units (counts, not flux) so they
    # get their own. No gamma here, since this synthetic scene has a narrow
    # dynamic range and a gamma would just wash it out - real captures are
    # photon-starved and usually do want one, which is why `uwbfan_recon.py`
    # writes its videos with gamma=1/2.2.
    photon_counts = bin_photon_stream(tt, 1 / (fps * T_exp))
    logger.info(f"  binned stream holds {photon_counts.mean():.2f} photons per voxel on average")
    write_video(
        flux_gt, output_folder / f"gt-{fps}fps.mp4",
        vmin=vmin, vmax=vmax, **vid_settings
    )
    write_video(
        photon_counts, output_folder / f"photons-{fps}fps.mp4",
        vmin=0, vmax=photon_counts.max(), **vid_settings
    )
    write_video(
        flux_recon, output_folder / f"fluxrecon-{fps}fps.mp4",
        vmin=vmin, vmax=vmax, **vid_settings
    )

    # Filmstrip of the first few frames, ground truth on top and reconstruction
    # below. Consecutive frames (rather than frames spread over the whole
    # window) make the point: what changes between adjacent frames is exactly
    # the motion this framerate is able to resolve.
    show_idxs = np.arange(4)
    fig, axes = plt.subplots(2, len(show_idxs), figsize=(3 * len(show_idxs), 6.5))
    for col, idx in enumerate(show_idxs):
        axes[0, col].imshow(flux_gt[idx], vmin=vmin, vmax=vmax, cmap="inferno")
        axes[0, col].set_title(f"t = {tt[idx] * T_exp * 1e3:.2f} ms")
        axes[1, col].imshow(flux_recon[idx], vmin=vmin, vmax=vmax, cmap="inferno")
        for row in range(2):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    axes[0, 0].set_ylabel("ground truth\n(band-limited)")
    axes[1, 0].set_ylabel("flux probing")
    fig.suptitle(f"{fps} fps reconstruction (Nyquist {nyquist:.0f} Hz), rel. RMSE {rel_rmse:.4f}")
    fig.tight_layout()
    fig.savefig(output_folder / f"frames-{fps}fps.png", dpi=120)
    plt.close(fig)

logger.info(f"Wrote spectrum, reconstructions, videos, and figures to {output_folder}")
