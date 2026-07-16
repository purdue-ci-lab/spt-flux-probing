import re

import finufft
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.stats import norm, normaltest
from tqdm import tqdm

from uwb3d import logger
from spadlib.sim import simulate_ihpp3d
import plotly.io as pio

pio.renderers.default = "browser"          # opens in your default browser tab
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times New Roman",
})

logger = logger.get_logger(__name__)


def flux_func_gaussian_ball_linear(
    T_exp,
    start_pos=(0.0, 1.0),   # (x, y) in normalised [0,1] coords, top-left
    end_pos=(1.0, 0.0),     # (x, y) in normalised [0,1] coords, bottom-right
    sigma=0.1,
    max_amplitude=1e4,
    normalize=False,
):
    def flux(x, y, t):
        # linear interpolation of centre position
        frac = t / T_exp if not normalize else t   # t is already in [0,1] if normalize
        curr_x = start_pos[0] + (end_pos[0] - start_pos[0]) * frac
        curr_y = start_pos[1] + (end_pos[1] - start_pos[1]) * frac
        return max_amplitude * np.exp(
            -((x - curr_x) ** 2 + (y - curr_y) ** 2) / (2 * sigma ** 2)
        )

    return flux, max_amplitude

T_exp = 1.0
GAUSS_LINEAR_BASE = dict(
    start_pos=(0.2, 0.2),
    end_pos=(0.8, 0.8),
    T_exp=T_exp,
    sigma=0.1,
    normalize=False
)

def mode_index(mode, shape):
    """
    Convert a signed-frequency mode tuple (ft, fy, fx) to
    the numpy array index produced by np.fft (output is
    arranged like np.fft.fftfreq, i.e. 0..N//2, -(N//2)...-1).
    NOT nufft
    """
    return tuple(int(m) % int(s) for m, s in zip(mode, shape))

def gaussian_contour_levels(mu, cov2, ax, n_sigma=(1, 2), color="cyan", lw=0.8):
    """Draw elliptical contours of a 2-D Gaussian on ax."""
    vals, vecs = np.linalg.eigh(cov2)
    vals = np.maximum(vals, 0.0)
    theta = np.linspace(0, 2 * np.pi, 300)
    max_sigma = max(n_sigma)
    base_rgb = np.array(mcolors.to_rgb(color))
    for ns in n_sigma:
        t = (ns - 1) / max_sigma                          # 0 at innermost, ~1 at outermost
        faded = base_rgb + (1 - base_rgb) * t * 0.7      # blend toward white
        ell = ns * np.stack([np.sqrt(vals[0]) * np.cos(theta),
                             np.sqrt(vals[1]) * np.sin(theta)])
        xy = vecs @ ell
        ax.plot(mu[0] + xy[0], mu[1] + xy[1],
                color=faded, lw=lw, linestyle="--")

# ======================================================
# Configuration
# ======================================================

# Amplitude levels to sweep (scales max_amplitude)
# hardcode attempt to keep E[nphotons] = 5, 50, 500, 5000
AMPLITUDE_LEVELS = (5 / 6.2) * np.array([1e2, 1e3 * (1/5), 1e3 * (2/5), 1e3, 1e4, 1e5])

N_TRIALS = 10000          # number of Monte-Carlo trials per amplitude level

# Fourier modes to probe (t-freq, y-freq, x-freq) — zero-based, centred convention
PROBE_MODES = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 2),
    (1, 0, 1),
    (1, 1, 0),
    (1, 2, 0),
    (2, 2, 2),
    (3, 1, 1),
    (5, 3, 7),
]

# Base Gaussian-ball args (max_amplitude will be overridden in the loop)
GAUSS_BASE = dict(
    height=1,
    width=1,
    x_dist=0.4,
    y_dist=0.4,
    T_exp=T_exp,
    hz_move_x=3.0,
    hz_move_y=0,
    hz_light=2.0,
    sigma=0.1,
    normalize=False,
)
GAUSS_LINEAR_BASE = dict(
    start_pos=(0.2, 0.2),
    end_pos=(0.8, 0.8),
    T_exp=T_exp,
    sigma=0.1,
    normalize=False
)

# --- Data caching ---------------------------------------------------
# If LOAD_DATA is True, skip the Monte-Carlo simulation and ground-truth
# integration and instead load the previously saved arrays from DATA_PATH.
# If SAVE_DATA is True, freshly computed results are written to DATA_PATH.
LOAD_DATA = True
SAVE_DATA = True
DATA_PATH = "output/data/normality_data.npz"

NUFFT_EPS = 1e-6

# Reconstruction grid sizes
n_modes = (12, 12, 16)  # we are not probing up to the resolution Nyquist
Nt, Ny, Nx = 10000, 100, 100
# tt, yy, xx = np.linspace(0, 1, Nt, endpoint=False), np.linspace(0, 1, Ny, endpoint=False), np.linspace(0, 1, Nx, endpoint=False)
# TT, YY, XX = np.meshgrid(tt, yy, xx, indexing="ij")

# fluxgt = ihpp_data.flux_func_gaussian_ball(**GAUSS_BASE, max_amplitude=1.0)[0](XX, YY, TT)
fluxfuncgt = flux_func_gaussian_ball_linear(**GAUSS_LINEAR_BASE, max_amplitude=1.0)[0]
# fluxgt = fluxfuncgt(XX, YY, TT)


# ======================================================
# Main simulation loop  (or load cached data)
# ======================================================

if LOAD_DATA:
    print(f"Loading cached data from {DATA_PATH}")
    with np.load(DATA_PATH, allow_pickle=False) as data:
        coeffs_arr       = data["coeffs"]        # (n_amps, n_modes, N_TRIALS) complex
        mean_photons_arr = data["mean_photons"]  # (n_amps, N_TRIALS)
        gt_params_arr    = data["gt_params"]     # (n_modes, 5)

    # Reconstruct the dict / list structures the plotting code expects.
    coeffs = {
        ai: {mi: coeffs_arr[ai, mi] for mi in range(len(PROBE_MODES))}
        for ai in range(len(AMPLITUDE_LEVELS))
    }
    mean_photons = {ai: mean_photons_arr[ai] for ai in range(len(AMPLITUDE_LEVELS))}
    gt_params = [tuple(row) for row in gt_params_arr]
else:
    # Storage: coefficients[amp_idx][mode_idx] = list of complex values
    coeffs = {
        ai: {mi: [] for mi in range(len(PROBE_MODES))}
        for ai in range(len(AMPLITUDE_LEVELS))
    }
    mean_photons = {ai: [] for ai in range(len(AMPLITUDE_LEVELS))}

    for ai, amp in enumerate(AMPLITUDE_LEVELS):
        print(f"Amplitude {amp:.0e}  ({ai+1}/{len(AMPLITUDE_LEVELS)})")

        # lam, lmax = ihpp_data.flux_func_gaussian_ball(
        #     max_amplitude=amp, **GAUSS_BASE
        # )
        lam, lmax = flux_func_gaussian_ball_linear(
            max_amplitude=amp, **GAUSS_LINEAR_BASE
        )

        for trial in tqdm(range(N_TRIALS)):
            points = simulate_ihpp3d(
                lambda_fn=lam,
                s_region=[[0, 0], [1, 0], [1, 1], [0, 1]],
                t_range=(0, 1),
                npoints=None,
                nx=100, ny=100, nt=100,
                lmax=lmax,
                rng=None,          # random seed each trial
            )

            n_photons = len(points)
            mean_photons[ai].append(n_photons)

            if n_photons == 0:
                # No photons: all coefficients are zero
                for mi in range(len(PROBE_MODES)):
                    coeffs[ai][mi].append(0.0 + 0.0j)
                continue

            px, py, pt = points[:, 0], points[:, 1], points[:, 2]

            F = finufft.nufft3d1(
                pt * 2 * np.pi,
                py * 2 * np.pi,
                px * 2 * np.pi,
                np.ones(n_photons, dtype="complex128"),
                n_modes=n_modes,
                eps=NUFFT_EPS,
                isign=-1,
            )

            for mi, mode in enumerate(PROBE_MODES):
                idx = mode_index(mode, F.shape)
                coeffs[ai][mi].append(np.fft.ifftshift(F)[idx])

        avg = np.mean(mean_photons[ai])
        print(f"  mean photons = {avg:.1f}")

    # Convert to arrays
    for ai in range(len(AMPLITUDE_LEVELS)):
        for mi in range(len(PROBE_MODES)):
            coeffs[ai][mi] = np.array(coeffs[ai][mi])


# ======================================================
# Plotting: grid of 2-D histograms (Re vs Im)
# ======================================================

# ------------------------------------------------------------------
# Pre-compute ground truth Gaussian parameters for each mode
# The flux grid TT, YY, XX and fluxgt must already be defined, e.g.:
#   xx = np.linspace(0, 1, Nx, endpoint=False); yy = np.linspace(0, 1, Ny, endpoint=False); tt = np.linspace(0, 1, Nt, endpoint=False)
#   TT, YY, XX = np.meshgrid(tt, yy, xx, indexing="ij")
#   fluxgt = fluxfuncgt(XX, YY, TT)          # shape (Nt, Ny, Nx)
#
# For amplitude sweeps the flux shape is proportional to amplitude, so
# we store a *normalised* version and scale per amplitude level below.
# ------------------------------------------------------------------

dvol = (1 / Nt) * (1 / Ny) * (1 / Nx)  # volume element of the grid; needed for normalisation of sums to integrals

if not LOAD_DATA:
    gt_params = []   # list over modes: (mu_re, mu_im, var_re, var_im, cov)
    # for mode in PROBE_MODES:
    #     ft, fy, fx = mode
    #     # Phase argument  2π f·x  evaluated on the full grid
    #     phase = 2 * np.pi * (ft * TT + fy * YY + fx * XX)
    #     cos_phase = np.cos(phase)
    #     sin_phase = np.sin(phase)

    #     # GT Fourier coefficient (mean of Re and Im)
    #     # E[NUFFT] = sum_x e^{-i2π f·x} λ(x) Δvol — a discrete sum
    #     mu_re_norm = np.sum(cos_phase  * fluxgt) * dvol
    #     mu_im_norm = np.sum(-sin_phase * fluxgt) * dvol

    #     # Second-moment sums (normalised)
    #     var_re_norm  = np.sum(cos_phase ** 2 * fluxgt) * dvol
    #     var_im_norm  = np.sum(sin_phase ** 2 * fluxgt) * dvol
    #     cov_norm     = np.sum(-cos_phase * sin_phase * fluxgt) * dvol

    #     gt_params.append((mu_re_norm, mu_im_norm,
    #                       var_re_norm, var_im_norm, cov_norm))

    # quad integrate instead of Riemann sum above (already has numerical error when
    # Gaussain ball starts moving)
    from scipy import integrate
    for mode in tqdm(PROBE_MODES):
        ft, fy, fx = mode

        def integrand_mu_re(t, y, x):
            phase = 2 * np.pi * (ft * t + fy * y + fx * x)
            return np.cos(phase) * fluxfuncgt(x, y, t)

        def integrand_mu_im(t, y, x):
            phase = 2 * np.pi * (ft * t + fy * y + fx * x)
            return -np.sin(phase) * fluxfuncgt(x, y, t)

        def integrand_var_re(t, y, x):
            phase = 2 * np.pi * (ft * t + fy * y + fx * x)
            return np.cos(phase)**2 * fluxfuncgt(x, y, t)

        def integrand_var_im(t, y, x):
            phase = 2 * np.pi * (ft * t + fy * y + fx * x)
            return np.sin(phase)**2 * fluxfuncgt(x, y, t)

        def integrand_cov(t, y, x):
            phase = 2 * np.pi * (ft * t + fy * y + fx * x)
            return -np.cos(phase) * np.sin(phase) * fluxfuncgt(x, y, t)

        ranges = [(0, 1), (0, 1), (0, 1)]   # t in [0,1], y in [0,1], x in [0,1]
        opts = {"limit": 50}

        mu_re,   _ = integrate.nquad(integrand_mu_re,  ranges, opts=opts)
        mu_im,   _ = integrate.nquad(integrand_mu_im,  ranges, opts=opts)
        var_re,  _ = integrate.nquad(integrand_var_re, ranges, opts=opts)
        var_im,  _ = integrate.nquad(integrand_var_im, ranges, opts=opts)
        cov,     _ = integrate.nquad(integrand_cov,    ranges, opts=opts)

        gt_params.append((mu_re, mu_im, var_re, var_im, cov))

    # ---- Save freshly computed data for later reuse ----
    if SAVE_DATA:
        import os
        os.makedirs(os.path.dirname(DATA_PATH) or ".", exist_ok=True)
        coeffs_arr = np.stack([
            np.stack([coeffs[ai][mi] for mi in range(len(PROBE_MODES))])
            for ai in range(len(AMPLITUDE_LEVELS))
        ])  # (n_amps, n_modes, N_TRIALS) complex
        mean_photons_arr = np.array(
            [mean_photons[ai] for ai in range(len(AMPLITUDE_LEVELS))]
        )  # (n_amps, N_TRIALS)
        gt_params_arr = np.array(gt_params)  # (n_modes, 5)
        np.savez(
            DATA_PATH,
            coeffs=coeffs_arr,
            mean_photons=mean_photons_arr,
            gt_params=gt_params_arr,
            amplitude_levels=np.asarray(AMPLITUDE_LEVELS),
            probe_modes=np.asarray(PROBE_MODES),
        )
        print(f"Saved simulation + ground-truth data to {DATA_PATH}")

n_amps  = len(AMPLITUDE_LEVELS)
n_modes = len(PROBE_MODES)
NBINS   = 40

for ai, amp in enumerate(AMPLITUDE_LEVELS):
    print(f"******************** Plotting amplitude {amp:.0e}  ({ai+1}/{n_amps}) ********************")
    for mi in range(n_modes):

        fig = plt.figure(figsize=(4, 4))
        inner_gs = gridspec.GridSpec(
            2, 2,
            figure=fig,
            width_ratios=[4, 1],
            height_ratios=[1, 4],
            hspace=0.0,
            wspace=0.0,
        )

        ax      = fig.add_subplot(inner_gs[1, 0])
        ax_top  = fig.add_subplot(inner_gs[0, 0], sharex=ax)
        ax_rgt  = fig.add_subplot(inner_gs[1, 1], sharey=ax)

        c = coeffs[ai][mi]
        re, im = c.real, c.imag

        # ---- ground truth params ----
        mu_re_n, mu_im_n, var_re_n, var_im_n, cov_n = gt_params[mi]
        mu_re, mu_im = mu_re_n * amp, mu_im_n * amp
        var_re, var_im = var_re_n * amp, var_im_n * amp
        cov = cov_n * amp

        # ---- define ranges centered on mu ----
        half_re = 4 * np.sqrt(var_re)
        half_im = 4 * np.sqrt(var_im)
        larger_halfvar = max(half_re, half_im)
        smaller_halfvar = min(half_re, half_im)
        # re_range = (mu_re - half_re, mu_re + half_re)
        # im_range = (mu_im - half_im, mu_im + half_im)
        # try to keep axis scales equal, and more easily show covariance, I think
        re_range = (mu_re - larger_halfvar, mu_re + larger_halfvar)
        im_range = (mu_im - larger_halfvar, mu_im + larger_halfvar)

        # ---- main 2D histogram ----
        ax.hist2d(re, im, bins=NBINS, cmap="gray", edgecolor='face', range=[re_range, im_range])

        print(f"Comparison for mode {PROBE_MODES[mi]}")
        print(f"\tGT mean (Re, Im) for mode {PROBE_MODES[mi]}: ({mu_re:.4f}, {mu_im:.4f})")
        print(f"\tGT var (Re, Im) and cov: var_re={var_re:.4f}, var_im={var_im:.4f}, cov={cov:.4f}")
        print(f"\tSimulated mean (Re, Im) for mode {PROBE_MODES[mi]}: ({re.mean():.4f}, {im.mean():.4f})")
        print(f"\tSimulated var (Re, Im) and cov: var_re={re.var():.4f}, var_im={im.var():.4f}, cov={np.cov(re, im)[0,1]:.4f}")

        # ---- normality test (D'Agostino-Pearson) on Re and Im marginals ----
        stat_re, p_re = normaltest(re)
        stat_im, p_im = normaltest(im)
        print(f"\tNormality test (Re): statistic={stat_re:.4f}, p-value={p_re:.4e}")
        print(f"\tNormality test (Im): statistic={stat_im:.4f}, p-value={p_im:.4e}")
        print(f"--------------------")

        # ---- Gaussian contours and GT mean ----
        mu   = np.array([mu_re, mu_im])
        cov2 = np.array([[var_re, cov],
                         [cov,    var_im]])

        gaussian_contour_levels(mu, cov2, ax, n_sigma=(1, 2, 3, 4), color="red", lw=0.8)
        ax.scatter(mu_re, mu_im, color="red", s=5, label="GT mean")

        # ---- Re marginal histogram (top) ----
        bins_re = np.linspace(re_range[0], re_range[1], NBINS + 1)
        counts_re, _ = np.histogram(re, bins_re)
        pdf_re = norm.pdf(bins_re[:-1], mu_re, np.sqrt(var_re))
        ax_top.bar(bins_re[:-1], counts_re / (counts_re.sum() * (bins_re[1] - bins_re[0])),
                   width=np.diff(bins_re), align="edge", color="steelblue", edgecolor='steelblue')
        ax_top.plot(bins_re[:-1], pdf_re, color="red", linewidth=1.2)

        # ---- Im marginal histogram (right) ----
        bins_im = np.linspace(im_range[0], im_range[1], NBINS + 1)
        counts_im, _ = np.histogram(im, bins_im)
        pdf_im = norm.pdf(bins_im[:-1], mu_im, np.sqrt(var_im))
        ax_rgt.barh(bins_im[:-1], counts_im / (counts_im.sum() * (bins_im[1] - bins_im[0])),
                    height=np.diff(bins_im), align="edge", color="steelblue", edgecolor='steelblue')
        ax_rgt.plot(pdf_im, bins_im[:-1], color="red", linewidth=1.2)

        # ---- cosmetics ----
        plt.setp(ax_top.get_xticklabels(), visible=False)
        plt.setp(ax_rgt.get_yticklabels(), visible=False)
        ax_top.tick_params(labelsize=5, left=False, labelleft=False, bottom=False, labelbottom=False)
        ax_rgt.tick_params(labelsize=5, bottom=False, labelbottom=False, left=False, labelleft=False)

        for spine in ax_top.spines.values():
            spine.set_visible(False)
        for spine in ax_rgt.spines.values():
            spine.set_visible(False)

        ax.xaxis.set_major_locator(MaxNLocator(nbins=2, prune=None))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=2, prune=None))

        ax.tick_params(labelsize=15)

        mode_str = str(PROBE_MODES[mi])
        avg_ph   = np.mean(mean_photons[ai])
        # ax_top.set_title(f"mode {mode_str}, amp={amp:.0e}, <N>={avg_ph:.0f}", fontsize=7)
        # ax.set_xlabel("Re", fontsize=7)
        # ax.set_ylabel("Im", fontsize=7)

        # ---- center on mu ----
        # half_re = 4 * np.sqrt(var_re)
        # half_im = 4 * np.sqrt(var_im)
        ax.set_xlim(re_range[0], re_range[1])
        ax.set_ylim(im_range[0], im_range[1])

        fname = f"output/plots/normality/normalitycont_ftfyfx{mode_str}_avgph{avg_ph:.0f}.svg"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.savefig(fname.replace(".svg", ".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
