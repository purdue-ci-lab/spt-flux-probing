from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sklearn.neighbors import KDTree
from tqdm.auto import tqdm


def fig_spt_points(points, r=0.1, max_pts=5000) -> go.Figure:
    """
    3D scatter plot of spatio-temporal points, colored by local density (number
    of neighbors within radius r).

    Args:
        points (np.ndarray): Array of shape (N, 3) with columns (x, y, t).
        r (float): Radius to count neighbors for density estimation.
    Returns:
        fig (go.Figure): 3D scatter plot with points colored by local density.
    """
    import visu3d as v3d
    if len(points) > max_pts:
        # reduce number of points for visualization
        idx = np.random.choice(len(points), size=max_pts, replace=False)
        points = points[idx]
    tree = KDTree(points[:, :3])
    neighbor_counts = tree.query_radius(points[:, :3], r=r, count_only=True)
    # Normalize neighbor_counts for colormap
    counts_norm = (neighbor_counts - neighbor_counts.min()) / (neighbor_counts.max() - neighbor_counts.min())
    cmap = plt.get_cmap("viridis")
    rgb = cmap(counts_norm)[:, :3]
    # # Normalize t to [0, 1] for colormap
    # t_norm = (t - t.min()) / (t.max() - t.min())
    # rgb = cm.viridis(t_norm)[:, :3]  # shape (N, 3), drop alpha
    trace = v3d.Point3d(p=points, rgb=rgb * 255).make_traces()[0]
    trace.update({"marker": {
        "showscale": True,
        "color": neighbor_counts,
        "colorscale": "Viridis",
        "colorbar": {"title": "Neighbor count within r=%.2f" % r, "len": 0.5, "y": 0.4},
    }})
    fig = v3d.make_fig([trace])
    fig.update_layout(title="Spatio-temporal events", scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="t"))
    return fig


def fig_powerspec(fx, fy, ft, F, step=1, maxpts=10000, threshold=None, func=None) -> go.Figure:
    """
    3D power spectrum of a Fourier transform.

    Args:
        fx, fy, ft: frequency axes (1D arrays)
        F: Fourier coefficients, shape (len(fx), len(fy), len(ft))
        step: subsampling step for visualization (e.g., 10 means show every 10th freq).
            Values within the steps are max-pooled.
    Returns:
        fig (go.Figure): 3D scatter plot of |F| on the (fx, fy, ft) grid, where color
            and size represent the magnitude.
    """
    import visu3d as v3d
    if func is None:
        func = np.abs
    # Subsample every step elements along each axis
    fx_sub = fx[::step]
    fy_sub = fy[::step]
    ft_sub = ft[::step]

    # Pad F so that its shape along each axis is divisible by step
    if step > 1:
        pad_x = (-F.shape[0]) % step
        pad_y = (-F.shape[1]) % step
        pad_t = (-F.shape[2]) % step
        F_padded = np.pad(F, ((0, pad_x), (0, pad_y), (0, pad_t)), mode='constant')

        # Max F over each step x step x step block
        new_shape = (
            F_padded.shape[0] // step, step,
            F_padded.shape[1] // step, step,
            F_padded.shape[2] // step, step
        )
        F_max = func(F_padded).reshape(new_shape).max(axis=(1, 3, 5))
    else:
        F_max = func(F)

    # Create meshgrid for the subsampled axes
    # ij indexing to match F shape
    xgrid, ygrid, tgrid = np.meshgrid(fx_sub, fy_sub, ft_sub, indexing="ij")
    if threshold is not None:
        mask = F_max > threshold
        F_max = F_max[mask]
        xgrid = xgrid[mask]
        ygrid = ygrid[mask]
        tgrid = tgrid[mask]
    if maxpts is not None and F_max.size > maxpts:
        print(f"Too many frequencies {F_max.size}.")
        return go.Figure()
    F_norm = (F_max - F_max.min()) / (F_max.max() - F_max.min())
    cmap = plt.get_cmap("viridis")
    colors = cmap(F_norm.reshape(-1))
    colors[:, 3] = F_norm.reshape(-1)
    colors = (colors[:, :4] * 255).astype(np.uint8)

    trace = v3d.Point3d(
        p=np.array([xgrid.reshape(-1), ygrid.reshape(-1), tgrid.reshape(-1)]).T,
        rgb=colors[:, :3]
    ).make_traces()[0]
    fig = v3d.make_fig(trace.update(
        marker=dict(
            size=40.0 * F_norm.reshape(-1),
            line=dict(
                width=0
            )
        )
    ))
    fig.update_layout(scene=dict(xaxis_title="freq x", yaxis_title="freq y", zaxis_title="freq t"))
    return fig


def ani_fluxvids(
        fluxvideos, playback_fps, titles=None, cmap="viridis", timestamps=None, constrained_layout=True,
        figsize=(4, 3), diverging=False, step=1
    ) -> FuncAnimation:
    """
    Each flux video is assumed to be of shape (time, height, width).

    Currently every video is assumed to have the same number of frames, and same FPS.

    Args:
        fluxvideos List[List[np.ndarray]]: [[(t x h x w) sequences]]
    """
    if isinstance(fluxvideos, np.ndarray) and fluxvideos.ndim == 3:
        fluxvideos = [[fluxvideos]]
    elif isinstance(fluxvideos, list) and all(isinstance(fv, np.ndarray) and fv.ndim == 3 for fv in fluxvideos):
        fluxvideos = [fluxvideos]
    interval = int(1000.0 / playback_fps)
    nrows = len(fluxvideos)
    ncols = max(len(row) for row in fluxvideos)
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows), constrained_layout=constrained_layout)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]
    ims = []
    ref_vid_len = None
    for i in range(nrows):
        imsrow = []
        for j in range(len(fluxvideos[i])):
            fluxvideo = fluxvideos[i][j]
            if fluxvideo is None:
                continue
            if ref_vid_len is None:
                ref_vid_len = fluxvideo.shape[0]
            if fluxvideo.shape[0] != ref_vid_len:
                raise ValueError("All flux videos must have the same number of frames")
            ax = axes[i][j]
            if isinstance(diverging, list):
                this_diverging = diverging[i][j]
            else:
                this_diverging = diverging
            if this_diverging:
                vmin = -max(abs(np.min(fluxvideo)), abs(np.max(fluxvideo)))
                vmax = max(abs(np.min(fluxvideo)), abs(np.max(fluxvideo)))
            else:
                vmin = np.min(fluxvideo)
                vmax = np.max(fluxvideo)
            if isinstance(cmap, list):
                this_cmap = cmap[i][j]
            else:
                this_cmap = cmap
            im = ax.imshow(fluxvideo[0], cmap=this_cmap, vmin=vmin, vmax=vmax)
            imsrow.append(im)
            if titles is not None:
                ax.set_title(titles[i][j])
            else:
                ax.set_title(f"Flux video {i},{j}")
            cbar = fig.colorbar(im, ax=ax, shrink=0.5)
            cbar.set_label("Flux (photons/sec/pixel^2)")
        ims.append(imsrow)

    if step > 1:
        ani_frames = ref_vid_len // step
    else:
        ani_frames = ref_vid_len
    def update(t):
        for i in range(nrows):
            for j in range(len(fluxvideos[i])):
                fluxvideo = fluxvideos[i][j]
                if fluxvideo is None:
                    continue
                im = ims[i][j]
                ax = axes[i][j]
                frameidx = t * step if step > 1 else t
                im.set_data(fluxvideo[frameidx])
                if timestamps is not None:
                    timestr = f"(frame={frameidx}, time={timestamps[frameidx]:.2f}s)"
                else:
                    timestr = f"(frame={frameidx})"
                if titles is not None:
                    ax.set_title(f"{titles[i][j]}\n{timestr}")
                else:
                    ax.set_title(f"Flux video {i},{j}\n{timestr}")
        return [im for row in ims for im in row]
    ani = FuncAnimation(fig, update, frames=ani_frames, interval=interval, blit=False)
    plt.close(fig)
    return ani


def fig_powerspec_spatial(F, fx=None, fy=None, bound=0.0, agg_fun=np.max, cmap="viridis", figsize=None):
    """
    Args:
        F: Either
            - Fourier coefficients with shape (len(ft), len(fy), len(fx)), or
            - 1D array of Fourier coefficients (len=N) where fx_coords and fy_coords
              give the fx, fy coordinate for each entry (multiple entries may share
              the same fx,fy because they correspond to different ft).
        fx: coordinates for the x-axis (frequency) or per-item fx for 1D F
        fy: coordinates for the y-axis (frequency) or per-item fy for 1D F
        agg_fun: function to aggregate over ft axis (e.g., np.max, np.sum). For the
                 1D-F case it will be applied to the set of values that share the same
                 (fx,fy) pair.
    """
    # Case A: full 3D array (ft, fy, fx) - behave as before
    if np.asarray(F).ndim == 3:
        absF = np.abs(F)
        # aggregate over ft axis -> shape (len(fy), len(fx))
        try:
            max_map = agg_fun(absF, axis=0)
        except TypeError:
            # some agg_fun definitions might not accept axis kw; fall back to calling on slices
            max_map = np.apply_along_axis(lambda a: agg_fun(a), 0, absF)

        # attempt to infer coords if not provided
        if fx is None or fy is None:
            # assume axes correspond to last two dims in order (fx, fy)
            fx = np.arange(max_map.shape[1]) if fx is None else fx
            fy = np.arange(max_map.shape[0]) if fy is None else fy

    # Case B: 1D array of coefficients with per-item fx,fy coordinates
    elif np.asarray(F).ndim == 1:
        F = np.asarray(F)
        if fx is None or fy is None:
            raise ValueError("fx_coords and fy_coords must be provided when F is 1D")
        fx_arr = np.asarray(fx)
        fy_arr = np.asarray(fy)
        if fx_arr.shape[0] != F.shape[0] or fy_arr.shape[0] != F.shape[0]:
            raise ValueError("fx_coords and fy_coords must have the same length as F when F is 1D")

        absF = np.abs(F)
        ux = np.unique(fx_arr)
        uy = np.unique(fy_arr)
        # grid shape should be (len(uy), len(ux)) to match (fy, fx) ordering used previously
        max_map = np.full((len(uy), len(ux)), -np.inf, dtype=float)

        # fill grid by aggregating values that share the same (fx, fy)
        for i_x, x in enumerate(ux):
            for i_y, y in enumerate(uy):
                idx = np.where((fx_arr == x) & (fy_arr == y))[0]
                if idx.size > 0:
                    vals = absF[idx]
                    try:
                        v = agg_fun(vals, axis=0)
                    except TypeError:
                        v = agg_fun(vals)
                    max_map[i_y, i_x] = v

        fx = ux
        fy = uy

    else:
        raise ValueError("F must be either a 3D array (ft, fy, fx) or a 1D array with per-item coords")

    # replace empty bins with NaN (they remain -inf from initialization)
    max_map[max_map == -np.inf] = np.nan
    max_map[max_map < bound] = np.nan

    # prepare for plotting (log10 to visualize wide dynamic range)
    eps = 1e-12
    log_map = np.log10(max_map + eps)

    if figsize is None:
        figsize = (9, 6)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        log_map,
        origin="lower",
        aspect="equal",
        extent=[float(fx[0]), float(fx[-1]), float(fy[0]), float(fy[-1])],
        cmap=cmap,
    )
    cbar = fig.colorbar(im)
    cbar.set_label("log10(max |F| over ft)")

    ax.set_title("Max amplitude over ft per (fx, fy)")
    ax.set_xlabel("fx")
    ax.set_ylabel("fy")
    return fig, ax


def fig_fourier_spectrum_3d(
    spectrum: np.ndarray,
    mode: str = "scatter",
    threshold_percentile: float = 80.0,
    colorscale: str = "Viridis",
    title: str = "3D Fourier Spectrum",
    is_rfft: bool = False,
) -> go.Figure:
    """
    Visualize a 3D Fourier spectrum using Plotly.
 
    Parameters
    ----------
    spectrum : np.ndarray
        Complex 3D array, shape (T, H, W) — time, spatial-y, spatial-x.
        Output of np.fft.fftn or np.fft.rfftn (see ``is_rfft``).
    mode : str
        'scatter' — scatter plot where dot size & opacity encode log magnitude.
        'voxel'   — volume/isosurface rendering where opacity encodes log magnitude.
    threshold_percentile : float
        Only points/voxels above this percentile of log-magnitude are shown.
        Useful to declutter the plot (default: 80 → top 20% shown).
    colorscale : str
        Plotly colorscale name (e.g. 'Viridis', 'Plasma', 'Turbo', 'Hot').
    title : str
        Figure title.
    is_rfft : bool
        If True, the spectrum was produced by rfftn with time as the real axis.
        Concretely: the caller did
            F = np.fft.rfftn(volume.transpose(1, 2, 0)).transpose(2, 0, 1)
        so the T axis (axis-0 of ``spectrum``) has length Nt//2 + 1 and covers
        non-negative temporal frequencies only [0, 1, …, Nt//2].
        The spatial axes (H, W) are always full complex FFTs.
 
    Returns
    -------
    plotly.graph_objects.Figure
    """
    if spectrum.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {spectrum.shape}")
 
    # ── Unpack in (T, H, W) order ─────────────────────────────────────────────
    Nt, Ny, Nx = spectrum.shape
 
    # ── Frequency axes ────────────────────────────────────────────────────────
    # Spatial axes: full FFT, shift so DC is at centre.
    fx = np.fft.fftshift(np.fft.fftfreq(Nx, d=1.0)) * Nx   # integer cycles
    fy = np.fft.fftshift(np.fft.fftfreq(Ny, d=1.0)) * Ny
 
    if is_rfft:
        # Real FFT along time: only non-negative frequencies, no shift needed.
        # rfftn output length along the real axis is Nt//2 + 1.
        # We recover the *original* Nt from the stored half-spectrum length.
        Nt_orig = 2 * (Nt - 1)         # inverse of rfft output length Nt//2+1
        ft = np.fft.rfftfreq(Nt_orig, d=1.0) * Nt_orig   # [0, 1, …, Nt_orig//2]
        shifted = np.fft.fftshift(spectrum, axes=(1, 2))  # shift spatial only
    else:
        Nt_orig = Nt
        ft = np.fft.fftshift(np.fft.fftfreq(Nt, d=1.0)) * Nt
        shifted = np.fft.fftshift(spectrum)               # shift all axes
 
    # ── Log magnitude ─────────────────────────────────────────────────────────
    magnitude = np.abs(shifted)
    log_mag = np.log1p(magnitude)           # log(1 + |F|) — safe for zeros
 
    # ── Threshold mask ────────────────────────────────────────────────────────
    thresh = np.percentile(log_mag, threshold_percentile)
    mask = log_mag >= thresh
 
    # ── Coordinate grids ──────────────────────────────────────────────────────
    # Array indexing is (T, H, W) = (Nt, Ny, Nx), so meshgrid "ij" order is:
    #   axis 0 → ft,  axis 1 → fy,  axis 2 → fx
    # Scene mapping: scene-x = fx, scene-y = fy, scene-z = ft
    FT, FY, FX = np.meshgrid(ft, fy, fx, indexing="ij")   # shapes (Nt, Ny, Nx)
 
    # ── Normalised log-magnitude [0, 1] for colour / opacity / size ───────────
    lm_flat = log_mag[mask]
    lm_min, lm_max = lm_flat.min(), lm_flat.max()
    lm_norm = (lm_flat - lm_min) / (lm_max - lm_min + 1e-12)
 
    if mode == "scatter":
        fig = _scatter_fig(
            FX[mask], FY[mask], FT[mask],   # scene (x, y, z) = (fx, fy, ft)
            lm_norm, lm_flat,
            colorscale,
        )
    elif mode == "voxel":
        fig = _volume_fig(
            FX, FY, FT, log_mag,
            thresh, colorscale,
        )
    else:
        raise ValueError(f"mode must be 'scatter' or 'voxel', got '{mode}'")
 
    # ── Axis labels / limits ──────────────────────────────────────────────────
    hx = Nx // 2
    hy = Ny // 2
    ht = Nt_orig // 2
 
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        scene=dict(
            xaxis=dict(
                title="f_x  (step=1)",
                range=[-hx, hx],
            ),
            yaxis=dict(
                title="f_y  (step=1)",
                range=[-hy, hy],
            ),
            zaxis=dict(
                title="f_t  (step=1)",
                range=[0, ht] if is_rfft else [-ht, ht],
            ),
            bgcolor="rgb(10,10,20)",
            xaxis_backgroundcolor="rgb(15,15,30)",
            yaxis_backgroundcolor="rgb(15,15,30)",
            zaxis_backgroundcolor="rgb(15,15,30)",
        ),
        paper_bgcolor="rgb(10,10,20)",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig
 
 
# ── Scatter helper ────────────────────────────────────────────────────────────
 
def _scatter_fig(x, y, z, lm_norm, lm_raw, colorscale):
    """Scatter3d: colour, opacity, and marker size all scale with log magnitude."""
    # Marker size: map [0,1] → [1, 20] px  (quadratic — emphasises peaks)
    sizes = 1 + (lm_norm ** 2) * 19
 
    # Per-point opacity: map [0,1] → [0.15, 1.0]
    opacities = 0.15 + lm_norm * 0.85
 
    import plotly.colors as pc
    rgb_list = pc.sample_colorscale(colorscale, lm_norm.tolist())
 
    def rgba(rgb_str, alpha):
        """'rgb(r,g,b)' → 'rgba(r,g,b,a)'"""
        inner = rgb_str[4:-1]
        return f"rgba({inner},{alpha:.3f})"
 
    colors = [rgba(rgb, a) for rgb, a in zip(rgb_list, opacities)]
 
    trace = go.Scatter3d(
        x=x.ravel(), y=y.ravel(), z=z.ravel(),
        mode="markers",
        marker=dict(
            size=sizes,
            color=lm_raw,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title="log(1+|F|)"),
            opacity=1.0,
            line=dict(width=0),
        ),
        text=[f"log|F|={v:.3f}" for v in lm_raw],
        hovertemplate="fx=%{x:.1f}  fy=%{y:.1f}  ft=%{z:.1f}<br>%{text}<extra></extra>",
    )
    # Per-point rgba overrides the uniform colorscale
    trace.marker.color = colors
    trace.marker.colorscale = None
    trace.marker.showscale = False
 
    fig = go.Figure(data=[trace])
    return fig
 
 
# ── Volume helper ─────────────────────────────────────────────────────────────
 
def _volume_fig(FX, FY, FT, log_mag, thresh, colorscale):
    """
    Volume rendering: opacity proportional to log magnitude.
    Uses go.Volume for smooth voxel-like rendering.
    Scene mapping: x=FX, y=FY, z=FT.
    """
    lm_max = log_mag.max()
 
    trace = go.Volume(
        x=FX.ravel(),
        y=FY.ravel(),
        z=FT.ravel(),
        value=log_mag.ravel(),
        isomin=thresh,
        isomax=lm_max,
        opacity=0.15,
        surface_count=20,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(title="log(1+|F|)"),
        caps=dict(x_show=False, y_show=False, z_show=False),
    )
    fig = go.Figure(data=[trace])
    return fig
