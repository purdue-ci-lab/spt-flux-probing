"""
Utilities for simulating inhomogeneous Poisson point processes (IHPP).
"""
from matplotlib.path import Path
import numpy as np
from tqdm import tqdm


# 1D inhomogenous point process functions
def simulate_ihpp1d(intensity_func, T, max_rate=None, seed=None):
    """
    Simulates a non-homogeneous Poisson process using the thinning algorithm.
    
    Args:
        intensity_func (callable): A function λ(t) defining the time-dependent intensity rate.
        T (float): The end time of the simulation (0 to T).
        max_rate (float): An upper bound for λ(t), the maximum intensity.
        device (str): Device to run the simulation ('cpu' or 'cuda').
    
    Returns:
        np.array: A 1D tensor of event times.
    """
    rng = np.random.default_rng(seed=seed)
    if max_rate is None:
        # Estimate max_rate by sampling the intensity function at a grid of points
        t_grid = np.linspace(0, T, 1000)
        max_rate = np.max(intensity_func(t_grid))
    # Generate homogeneous Poisson events
    n_events = int(2 * T * max_rate)  # Overestimate the number of events
    u = rng.random(n_events)
    inter_arrival_times = -np.log(u) / max_rate
    event_times = np.cumsum(inter_arrival_times)
    
    # Keep only those within [0, T]
    event_times = event_times[event_times <= T]
    
    # Thinning step
    acceptance_probs = rng.random(event_times.shape[0])
    accepted = acceptance_probs <= (intensity_func(event_times) / max_rate)
    
    return event_times[accepted]


def simulate_ihpp1d_torch(intensity_func, T, max_rate, device="cpu"):
    """
    simulate_ihpp1d except torch. Will crash if used with finufft for some reason.
    """
    # put in here so it doesn't crash finufft
    import torch
    # max_rate = 2E6
    # Generate homogeneous Poisson events
    n_events = int(2 * T * max_rate)  # Overestimate the number of events
    u = torch.rand(n_events, device=device)
    inter_arrival_times = -torch.log(u) / max_rate
    event_times = torch.cumsum(inter_arrival_times, dim=0)
    
    # Keep only those within [0, T]
    event_times = event_times[event_times <= T]
    
    # Thinning step
    acceptance_probs = torch.rand(event_times.size(0), device=device)
    accepted = acceptance_probs <= (intensity_func(event_times) / max_rate)
    
    return event_times[accepted]


def deadtime_drop_1d(timestamps, deadtime):
    """
    Applies a dead-time effect to a sequence of event timestamps by dropping events
    that occur within `deadtime` of the previous accepted event.
    
    Args:
        timestamps (np.array): A 1D array of event timestamps (assumed sorted).
        deadtime (float): The minimum time interval required between accepted events.
    
    Returns:
        np.array: A 1D array of timestamps after applying the dead-time effect.
    """
    if len(timestamps) == 0:
        return timestamps
    accepted = [timestamps[0]]
    for t in timestamps[1:]:
        if t - accepted[-1] >= deadtime:
            accepted.append(t)
    return np.array(accepted)


# 2D/3D inhomogenous point process functions
# ---------- helpers ----------
def _make_grid(poly, nx=100, ny=100):
    """
    Build a regular grid over the polygon's bounding box, return cell centers,
    increments, and a boolean mask of cells whose centers fall inside `poly`.
    poly: (K,2) array-like with vertices [x,y] (closed or open)
    """
    poly = np.asarray(poly, float)
    if not np.all(poly[0] == poly[-1]):
        poly = np.vstack([poly, poly[0]])  # close polygon if needed

    xmin, ymin = poly[:-1].min(axis=0)
    xmax, ymax = poly[:-1].max(axis=0)

    xinc = (xmax - xmin) / nx
    yinc = (ymax - ymin) / ny
    xgrid = xmin + (np.arange(nx) + 0.5) * xinc
    ygrid = ymin + (np.arange(ny) + 0.5) * yinc

    xx = np.repeat(xgrid[:, None], ny, axis=1)
    yy = np.repeat(ygrid[None, :], nx, axis=0)

    # mask: which grid centers are inside polygon
    P = Path(poly)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    inside = P.contains_points(pts)
    mask = inside.reshape(nx, ny)

    return dict(xgrid=xgrid, ygrid=ygrid, xx=xx, yy=yy,
                xinc=xinc, yinc=yinc, mask=mask, poly=poly)


def _sample_uniform_in_poly(N, poly, rng):
    """Rejection sample N uniform (x,y) inside polygon `poly`."""
    poly = np.asarray(poly, float)
    if not np.all(poly[0] == poly[-1]):
        poly = np.vstack([poly, poly[0]])
    path = Path(poly)
    xmin, ymin = poly[:-1].min(axis=0)
    xmax, ymax = poly[:-1].max(axis=0)

    out_x, out_y = [], []
    # sample in batches to avoid many small loops
    while len(out_x) < N:
        B = max(1000, int(1.3 * (N - len(out_x))))
        xs = rng.uniform(xmin, xmax, size=B)
        ys = rng.uniform(ymin, ymax, size=B)
        keep = path.contains_points(np.column_stack([xs, ys]))
        out_x.extend(xs[keep].tolist())
        out_y.extend(ys[keep].tolist())
    return np.asarray(out_x[:N]), np.asarray(out_y[:N])


# ---------- main simulator ----------
def simulate_ihpp2d(
    lambda_fn,
    s_region=None,
    npoints=None,
    nx=100, ny=100,
    lmax=None,
    rng=None,
    return_meta=False
):
    """
    Inhomogeneous Poisson process on a polygon (2D) via global thinning.

    Parameters
    ----------
    lambda_fn : callable
        Function f(x, y) -> nonnegative intensity.
    s_region : array-like (K,2)
        Polygon vertices (closed or open). Defaults to unit square.
    npoints : int or None
        If None, draw N ~ Poisson(∫∫ λ dx dy). If given, keep simulating until at least
        npoints are accepted, then truncate to exactly npoints.
    nx, ny : int
        Grid resolution used to estimate ∫ λ and (optionally) lmax if lmax is None.
    lmax : float or None
        Known upper bound of λ over the domain. If None, estimated from the grid.
    rng : int or numpy.random.Generator
    return_meta : bool
        If True, also returns dict of metadata (estimated mean, lmax, etc.).

    Returns
    -------
    points : (N,2) ndarray
        Accepted points as columns [x, y].
    meta : dict (optional)
        Info: lmax, mean_count_estimate, area, grid arrays, etc.
    """
    rng = np.random.default_rng(rng)
    if s_region is None:
        s_region = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]], float)
    else:
        s_region = np.asarray(s_region, float)
        if not np.all(s_region[0] == s_region[-1]):
            s_region = np.vstack([s_region, s_region[0]])

    # build grid & evaluate lambda to estimate integral and lmax if needed
    grid = _make_grid(s_region, nx=nx, ny=ny)
    flat_x = grid["xx"].ravel()
    flat_y = grid["yy"].ravel()
    vals = lambda_fn(flat_x, flat_y)
    vals = np.asarray(vals, float)
    M = vals.reshape(nx, ny)
    M[~grid["mask"]] = np.nan
    # Riemann sum estimate of integral over spatial domain
    mean_count_est = np.nansum(M) * grid["xinc"] * grid["yinc"]

    # choose lmax
    if lmax is None:
        lmax = float(np.nanmax(M))
        if not np.isfinite(lmax):
            raise ValueError("Failed to estimate lmax; check lambda_fn or region.")
    if lmax <= 0:
        raise ValueError("lmax must be positive.")

    # Spatial area via polygon shoelace
    poly = grid["poly"][:-1]
    area = 0.5 * np.abs(np.sum(poly[:,0]*np.roll(poly[:,1], -1) - poly[:,1]*np.roll(poly[:,0], -1)))

    points = []
    target = npoints if npoints is not None else None

    def _propose_accept(batch_N):
        xs, ys = _sample_uniform_in_poly(batch_N, s_region, rng)
        lam = lambda_fn(xs, ys)
        lam = np.asarray(lam, float)
        if np.any(lam < 0):
            raise ValueError("lambda_fn returned negative intensity.")
        keep = rng.random(batch_N) < (lam / lmax)
        return np.column_stack([xs[keep], ys[keep]])

    if target is None:
        N_prop = rng.poisson(lmax * area)
        if N_prop > 0:
            accepted = _propose_accept(N_prop)
            points.append(accepted)
    else:
        while sum(len(p) for p in points) < target:
            chunk = max(1000, int(1.2 * (target - sum(len(p) for p in points))))
            points.append(_propose_accept(chunk))
        P = np.vstack(points)
        if len(P) > target:
            idx = rng.choice(len(P), size=target, replace=False)
            P = P[idx]
        points = [P]

    P = np.vstack(points) if points else np.empty((0, 2))

    if return_meta:
        meta = dict(
            lmax=lmax,
            mean_count_estimate=float(mean_count_est),
            area=float(area),
            grid=dict(xgrid=grid["xgrid"], ygrid=grid["ygrid"],
                        xinc=grid["xinc"], yinc=grid["yinc"],
                        mask=grid["mask"])
        )
        return P, meta
    return P


def simulate_ihpp3d(
    lambda_fn,
    s_region=None,
    t_range=(0.0, 1.0),
    npoints=None,
    nx=100, ny=100, nt=100,
    lmax=None,
    rng=None,
    return_meta=False
):
    """
    Inhomogeneous Poisson process on a polygon x time, via global thinning.

    Parameters
    ----------
    lambda_fn : callable
        Function f(x, y, t) -> nonnegative intensity.
    s_region : array-like (K,2)
        Polygon vertices (closed or open). Defaults to unit square.
    t_range : (tmin, tmax)
        Time window.
    npoints : int or None
        If None, draw N ~ Poisson(∫∫∫ λ dx dy dt). If given, keep simulating
        until at least N are accepted, then truncate to exactly N.
    nx, ny, nt : int
        Grid resolution used only to estimate ∫ λ and (optionally) lmax if lmax is None.
    lmax : float or None
        Known upper bound of λ over the domain. If None, estimated from the grid.
    rng : int or numpy.random.Generator
    return_meta : bool
        If True, also returns dict of metadata (estimated mean, lmax, etc.).

    Returns
    -------
    points : (N,3) ndarray
        Accepted points as columns [x, y, t].
    meta : dict (optional)
        Info: lmax, mean_count_estimate, grid arrays, etc.
    """
    rng = np.random.default_rng(rng)
    if s_region is None:
        s_region = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]], float)
    else:
        s_region = np.asarray(s_region, float)
        if not np.all(s_region[0] == s_region[-1]):
            s_region = np.vstack([s_region, s_region[0]])

    tmin, tmax = float(t_range[0]), float(t_range[1])
    if tmax <= tmin:
        raise ValueError("t_range must have tmax > tmin")

    # Build grid & evaluate lambda to estimate integral and lmax if needed
    grid = _make_grid(s_region, nx=nx, ny=ny)
    times = np.linspace(tmin, tmax, nt)
    # Lambda cube on the grid (masked outside poly)
    Lambda = np.full((nx, ny, nt), np.nan, float)
    flat_x = grid["xx"].ravel()
    flat_y = grid["yy"].ravel()
    for it, tt in enumerate(times):
        vals = lambda_fn(flat_x, flat_y, np.full_like(flat_x, tt))
        M = vals.reshape(nx, ny)
        M[~grid["mask"]] = np.nan
        Lambda[:, :, it] = M

    # estimate integral ∫ λ dx dy dt using Riemann sum
    tinc = (tmax - tmin) / (nt - 1) if nt > 1 else (tmax - tmin)
    mean_count_est = np.nansum(Lambda) * grid["xinc"] * grid["yinc"] * tinc

    # choose lmax
    if lmax is None:
        lmax = float(np.nanmax(Lambda))
        if not np.isfinite(lmax):
            raise ValueError("Failed to estimate lmax; check lambda_fn or region.")
    if lmax <= 0:
        raise ValueError("lmax must be positive.")

    # Spatial area via polygon shoelace
    poly = grid["poly"][:-1]
    area = 0.5 * np.abs(np.sum(poly[:,0]*np.roll(poly[:,1], -1) - poly[:,1]*np.roll(poly[:,0], -1)))
    volume = area * (tmax - tmin)

    points = []
    target = npoints if npoints is not None else None

    def _propose_accept(batch_N):
        # propose (x,y,t) uniformly over the domain
        xs, ys = _sample_uniform_in_poly(batch_N, s_region, rng)
        ts = rng.uniform(tmin, tmax, size=batch_N)
        lam = lambda_fn(xs, ys, ts)
        if np.any(lam < 0):
            raise ValueError("lambda_fn returned negative intensity.")
        keep = rng.random(batch_N) < (lam / lmax)
        return np.column_stack([xs[keep], ys[keep], ts[keep]])

    if target is None:
        # Single Poisson draw
        N_prop = rng.poisson(lmax * volume)
        if N_prop > 0:
            accepted = _propose_accept(N_prop)
            points.append(accepted)
    else:
        # Keep drawing until we reach target; truncate to exactly npoints
        while sum(len(p) for p in points) < target:
            # propose in reasonably large chunks proportional to expected count
            chunk = max(1000, int(1.2 * (target - sum(len(p) for p in points))))
            points.append(_propose_accept(chunk))
        P = np.vstack(points)
        if len(P) > target:
            # random subset to match npoints exactly
            idx = rng.choice(len(P), size=target, replace=False)
            P = P[idx]
        points = [P]

    P = np.vstack(points) if points else np.empty((0, 3))

    if return_meta:
        meta = dict(
            lmax=lmax,
            mean_count_estimate=float(mean_count_est),
            area=float(area),
            volume=float(volume),
            grid=dict(xgrid=grid["xgrid"], ygrid=grid["ygrid"], times=times,
                      xinc=grid["xinc"], yinc=grid["yinc"], tinc=tinc,
                      mask=grid["mask"])
        )
        return P, meta
    return P
