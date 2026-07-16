import itertools

import matplotlib.patches as patches_l
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import f
from tqdm import tqdm
import torch


def normalize_velocities(velocities: np.ndarray, temporal_samples: int, spatial_samples: int):
    """
    Convert velocity bounds from pixel-per-frame units to a dimensionless normalized slope.

    This maps velocities defined on the discrete (x, t) grid to velocities in normalized
    coordinates x' = x / num_x and t' = t / num_t:

        v_norm = v_px_per_frame * (num_t / num_x)
               = (Δx/Δt) * (num_t/num_x)
               = (Δx/num_x) / (Δt/num_t)

    Parameters
    ----------
    velocities : np.ndarray
        Velocity bounds expressed in pixels per frame (Δx / Δt).
        Can be shape (2,) for [v_min, v_max] or any array of bounds.
    temporal_samples : int
        Number of frames (temporal samples) in the sequence.
    spatial_samples: int
        Number of pixels (spatial samples).

    Returns
    -------
    np.ndarray
        Velocity bounds in normalized units (dimensionless), i.e. slope in (x', t') space.
    """

    return velocities * (temporal_samples / spatial_samples)


def suppress_static_band(image, frequency, dc_bins=1, baseline_percentile=10.0):
    """
    Suppress near-DC components along the first axis and remove a column-wise baseline
    from a 2D frequency-frequency power plane.

    `image` is typically a projected power spectrum plane such as
    (ft x fx) or (ft x fy), obtained by summing |F(ft, fy, fx)|^2 over one spatial
    frequency dimension. Passing `frequency=ft` removes the temporal-DC band (ft ≈ 0),
    which corresponds to static / near-static content over time.

    Steps
    -----
    1) Zero out a narrow band around frequency ≈ 0 along the first axis:
         image[|frequency| < dc_bins * Δf, :] = 0
    2) Subtract a column-wise low-percentile baseline (estimated across the first axis),
       then clamp negatives to zero.

    Parameters
    ----------
    image : np.ndarray
        2D array of shape (len(frequency), M), e.g. a (ft x fx) or (ft x fy) power plane.
    frequency : np.ndarray
        1D array of bin centers corresponding to the first axis of `image`.
    dc_bins : int, optional
        Width of the suppressed near-DC band in units of frequency bins (default 1).
    baseline_percentile : float, optional
        Percentile used to estimate the per-column baseline across the first axis
        (default 10.0).

    Returns
    -------
    np.ndarray
        Processed 2D plane with near-DC rows removed and column-wise baseline suppressed.
    """

    image = image.copy()

    # Remove frequency ≈ 0
    delta_ft = abs(frequency[1] - frequency[0])
    dc_band = np.abs(frequency) < dc_bins * delta_ft
    image[dc_band, :] = 0

    # Column-wise low-percentile baseline subtraction
    baseline = np.percentile(image, baseline_percentile, axis=0, keepdims=True)
    image = image - baseline
    image[image < 0] = 0

    return image

def suppress_static_band_3d(F2: np.ndarray, ft: np.ndarray, dc_bins: int, baseline_percentile: float):
    """
    Minimal 3D version:
    - zero a band around ft=0 (or around center if ft is fftshifted)
    - subtract a baseline per-(fy,fx) column using a percentile across time
    """
    A = F2.copy()

    # zero near ft=0
    idx0 = int(np.argmin(np.abs(ft)))
    lo = max(0, idx0 - dc_bins)
    hi = min(A.shape[0], idx0 + dc_bins + 1)
    A[lo:hi, :, :] = 0

    # percentile baseline across time for each spatial freq bin (fy,fx)
    base = np.percentile(A, baseline_percentile, axis=0, keepdims=True)  # (1,Ny,Nx)
    A = A - base
    A[A < 0] = 0
    return A

def compute_energies_on_single_component_velocity_planes(
    velocities_px_frame: tuple[float, float, float],
    ft: np.ndarray,
    fy: np.ndarray,
    fx: np.ndarray,
    F: np.ndarray,
    num_t: int,
    num_x: int,
    direction: str = "x",
    dc_bins: int = 1,
    baseline_percentile: float = 10.0,
    sigma: float = 0.2,
):
    """
    Compute energy responses along single-component velocity planes in the 3D power spectrum.

    The 3D spectrum |F(ft, fy, fx)|² is projected onto either the (ft, fx) or (ft, fy) plane,
    static (ft ≈ 0) components are suppressed, and energy is accumulated along
    velocity-aligned lines for a set of candidate velocities.

    Parameters
    ----------
    velocities_px_frame : tuple of float
        (v_min, v_max, num_v) velocity range in pixels per frame.
    ft, fy, fx : np.ndarray
        Temporal and spatial frequency axes corresponding to F.
    F : np.ndarray
        3D Fourier transform with shape (len(ft), len(fy), len(fx)).
    num_t : int
        Number of temporal samples.
    num_x : int
        Number of spatial samples along the selected direction.
    direction : {"x", "y"}, optional
        Spatial component used for velocity evaluation (default "x").
    dc_bins : int, optional
        Width of the suppressed near-DC temporal band (default 1).
    baseline_percentile : float, optional
        Percentile for column-wise baseline subtraction (default 10.0).
    sigma : float, optional
        Width of the Gaussian mask orthogonal to the velocity line.

    Returns
    -------
    np.ndarray
        Energy values for each tested velocity.
    """
        
    # Velocity grid
    if len(velocities_px_frame) != 3:
        raise ValueError("velocities_px_frame must have length 3")
    v_bounds_px = np.linspace(*velocities_px_frame)
    v_bounds_norm = normalize_velocities(v_bounds_px, num_t, num_x)
    # Power spectrum
    F2 = np.abs(F) ** 2
    energies = []
    # Select projection plane
    if direction == "x":
        F2_plane = F2.sum(axis=1)
        F2_plane = suppress_static_band(F2_plane, ft, dc_bins, baseline_percentile)
        freq_axis = fx
    else:
        F2_plane = F2.sum(axis=2)
        F2_plane = suppress_static_band(F2_plane, ft, dc_bins, baseline_percentile)
        freq_axis = fy

    for v in v_bounds_norm:
        FT, FX = np.meshgrid(ft, freq_axis, indexing="ij")
        norm = np.sqrt(1 + v * v)
        d = (FT + v * FX) / norm
        # define a line of coefficients
        mask = np.exp(-(d**2) / (2 * sigma**2))
        hH, hW = mask.shape
        # Pad template to image size
        pad = np.zeros_like(F2_plane)
        pad[:hH, :hW] = mask

        energies.append(np.sum(F2_plane * pad))

    return np.array(energies)

def compute_energies_on_multiple_component_velocity_planes(
    velocities_px_frame,
    ft, fy, fx, F,
    num_t, num_x, num_y=None,
    dc_bins=1,
    baseline_percentile=10.0,
    sigma=0.2,
):
    if num_y is None:
        num_y = len(fy)

    # velocity grid
    v_px = np.linspace(*velocities_px_frame)
    vx = normalize_velocities(v_px, num_t, num_x)
    vy = normalize_velocities(v_px, num_t, num_y)
    Nv = len(vx)

    # power spectrum
    F2 = np.abs(F)**2
    F2s = suppress_static_band_3d(F2, ft, dc_bins, baseline_percentile)

    # flatten coordinates
    FT, FY, FX = np.meshgrid(ft, fy, fx, indexing="ij")
    P  = F2s.ravel()
    ft_f = FT.ravel()
    fx_f = FX.ravel()
    fy_f = FY.ravel()

    energies = np.zeros((Nv, Nv), dtype=np.float64)

    # vectorized over vy
    for i, vxi in tqdm(enumerate(vx),total=len(vx)):
        base = ft_f + vxi * fx_f                 # (N,)
        xi_all = base[:, None] + fy_f[:, None] * vy[None, :]  # (N, Nv)

        mask = np.abs(xi_all) < sigma            # boolean (N, Nv)
        energies[i, :] = (P[:, None] * mask).sum(axis=0)

    return energies

def compute_energies_velocity_planes_torch(
    velocities_px_frame,
    ft, fy, fx,
    F,
    num_t, num_x, num_y=None,
    dc_bins=1,
    baseline_percentile=10.0,
    sigma=0.2,
    device="cuda",
    keep_frac=0.01,        # keep top 1% coefficients; set None to disable
    max_keep=5_000_000,    # safety cap to avoid insane memory
):
    if num_y is None:
        num_y = len(fy)

    # ---- velocity grid (normalize on CPU; tiny) ----
    v_px = np.linspace(*velocities_px_frame)  # (Nv,)
    vx = torch.tensor(normalize_velocities(v_px, num_t, num_x), device=device, dtype=torch.float32)
    vy = torch.tensor(normalize_velocities(v_px, num_t, num_y), device=device, dtype=torch.float32)
    Nv = vx.numel()

    # ---- power spectrum ----
    F_t = torch.as_tensor(F, device=device)
    F2 = torch.abs(F_t) ** 2

    # suppression on CPU once (keeps your existing implementation)
    F2s = torch.tensor(
        suppress_static_band_3d(F2.detach().cpu().numpy(), ft, dc_bins, baseline_percentile),
        device=device,
        dtype=torch.float32,
    )

    # ---- frequency grids ----
    FT, FY, FX = torch.meshgrid(
        torch.tensor(ft, device=device, dtype=torch.float32),
        torch.tensor(fy, device=device, dtype=torch.float32),
        torch.tensor(fx, device=device, dtype=torch.float32),
        indexing="ij",
    )

    # ---- flatten ----
    P    = F2s.reshape(-1)
    ft_f = FT.reshape(-1)
    fx_f = FX.reshape(-1)
    fy_f = FY.reshape(-1)

    # ---- pruning: top-k (no quantile) ----
    if keep_frac is not None:
        N = P.numel()
        K = int(N * float(keep_frac))
        K = max(1, min(K, int(max_keep), N))

        _, idx = torch.topk(P, k=K, largest=True, sorted=False)
        P    = P[idx]
        ft_f = ft_f[idx]
        fx_f = fx_f[idx]
        fy_f = fy_f[idx]

    # ---- accumulate energies ----
    energies = torch.zeros((Nv, Nv), device=device, dtype=torch.float32)

    for i in tqdm(range(Nv)):
        base = ft_f + vx[i] * fx_f              # (K,)
        xi = base[:, None] + fy_f[:, None] * vy[None, :]  # (K, Nv)
        mask = torch.abs(xi) < sigma
        energies[i] = (P[:, None] * mask).sum(dim=0)

    return energies.detach().cpu().numpy()   

def compute_energies_velocity_planes_torch_batched(
    velocities_px_frame,
    ft, fy, fx,
    F,
    num_t, num_x, num_y=None,
    dc_bins=1,
    baseline_percentile=10.0,
    sigma=0.2,
    device="cuda",
    keep_frac=0.01,        # keep top 1% coefficients
    max_keep=5_000_000,    # safety cap
    batch_size=32,         # NEW: Batches the velocity grid computation
):
    if num_y is None:
        num_y = len(fy)

    # 1. ---- Setup Grids (CPU) ----
    # Normalize velocities
    v_px = np.linspace(*velocities_px_frame)
    vx_cpu = normalize_velocities(v_px, num_t, num_x)
    vy_cpu = normalize_velocities(v_px, num_t, num_y)
    
    # Create frequency grids on CPU first to save VRAM
    FT_cpu, FY_cpu, FX_cpu = np.meshgrid(ft, fy, fx, indexing="ij")
    
    # 2. ---- Power Spectrum & Suppression (CPU ONLY) ----
    # Fix: Do not move full F to GPU. Calculate power on CPU to avoid immediate OOM.
    # Note: We use np.abs(F)**2. If F is huge, ensure you have enough System RAM.
    # print("Computing power spectrum on CPU...")
    if torch.is_tensor(F):
        F = F.detach().cpu().numpy()
        
    F2_cpu = np.abs(F) ** 2
    
    # Run suppression on CPU (as originally intended)
    # print("Suppressing static band...")
    F2s_cpu = suppress_static_band_3d(F2_cpu, ft, dc_bins, baseline_percentile)
    
    # Flatten everything on CPU
    P_flat = F2s_cpu.reshape(-1)
    ft_flat = FT_cpu.reshape(-1)
    fx_flat = FX_cpu.reshape(-1)
    fy_flat = FY_cpu.reshape(-1)

    # 3. ---- Pruning / Top-K (CPU) ----
    # We prune BEFORE going to GPU. This is the key to fixing the OOM.
    if keep_frac is not None:
        # print(f"Pruning to top {keep_frac*100}%...")
        N = P_flat.size
        K = int(N * float(keep_frac))
        K = max(1, min(K, int(max_keep), N))

        # Use np.argpartition for fast top-k selection on CPU
        # (partition puts the K-th element in place, everything larger to the right)
        idx = np.argpartition(P_flat, -K)[-K:]
        
        # Filter arrays
        P_flat = P_flat[idx]
        ft_flat = ft_flat[idx]
        fx_flat = fx_flat[idx]
        fy_flat = fy_flat[idx]

    # 4. ---- Transfer to GPU ----
    # print(f"Moving {len(P_flat)} elements to {device}...")
    P_gpu = torch.as_tensor(P_flat, device=device, dtype=torch.float32)
    ft_gpu = torch.as_tensor(ft_flat, device=device, dtype=torch.float32)
    fx_gpu = torch.as_tensor(fx_flat, device=device, dtype=torch.float32)
    fy_gpu = torch.as_tensor(fy_flat, device=device, dtype=torch.float32)
    
    vx_gpu = torch.as_tensor(vx_cpu, device=device, dtype=torch.float32)
    vy_gpu = torch.as_tensor(vy_cpu, device=device, dtype=torch.float32)
    
    Nv = len(vx_gpu)
    
    # 5. ---- Batch Process Energies ----
    # If we compute (K, Nv) all at once, we might OOM if K is large.
    # We batch the inner 'vy' loop.
    
    energies = torch.zeros((Nv, Nv), device=device, dtype=torch.float32)

    # Outer loop: Iterates over X velocities (Rows of energy matrix)
    for i in range(Nv):
        # Calculate base term for this specific vx[i]
        # Shape: (K,)
        base = ft_gpu + vx_gpu[i] * fx_gpu
        base = base[:, None] # Shape: (K, 1)

        # Inner Batching: Process Y velocities in chunks to save memory
        # We need to fill energies[i, :] which has length Nv
        for j_start in range(0, Nv, batch_size):
            j_end = min(j_start + batch_size, Nv)
            
            # Slice of vy for this batch
            vy_batch = vy_gpu[j_start:j_end] # Shape: (B,)
            
            # xi shape: (K, B)
            # This is the heavy matrix. By limiting B (batch_size), we limit VRAM usage.
            xi = base + fy_gpu[:, None] * vy_batch[None, :]
            
            # Mask and Sum
            mask = torch.abs(xi) < sigma
            
            # Sum over K (dim 0), resulting in (B,)
            batch_energy = (P_gpu[:, None] * mask).sum(dim=0)
            
            # Assign to result matrix
            energies[i, j_start:j_end] = batch_energy

    return energies.detach().cpu().numpy()

def f_cfar_general(
    energies,
    alpha,
    window=5,
    guard=1,
    M=1,
    nu_cell=2,
):
    """
    F-distribution CFAR (F-CFAR) with generic chi-square DOF per cell.

    Each energy cell E[i] is assumed ~ chi^2_{nu_cell} (up to a scale).
    Then:
        X = sum of M CUT cells ~ chi^2_{nu_cell * M}
        Y = sum of N training cells ~ chi^2_{nu_cell * N}

    Test statistic:
        T = (X / (nu_cell * M)) / (Y / (nu_cell * N))
          = (X * N) / (Y * M)
      and
        T ~ F_{nu_cell * M, nu_cell * N} under H0.

    Parameters
    ----------
    energies : array-like
        Input energy values (already integrated/summed).
    alpha : float
        Desired false-alarm probability Pfa.
    window : int
        Number of cells on each side of CUT.
    guard : int
        Guard cells excluded on each side of CUT.
    M : int
        Number of CUT cells averaged (default 1: single-bin CUT).
    nu_cell : int or float
        Chi-square DOF of a *single* energy cell, i.e. 2 * (#coeffs summed).

    Returns
    -------
    T : ndarray
        F-type test statistic per CUT.
    E_bg : ndarray
        Background mean estimate (for plotting/interpretation).
    detections : ndarray
        Boolean detection mask.
    thresholds : ndarray
        Per-CUT F threshold (can vary near edges where N_i changes).
    """

    E = np.array(energies, dtype=float)
    n = len(E)

    T = np.zeros_like(E, dtype=float)
    E_bg = np.zeros_like(E, dtype=float)
    detections = np.zeros_like(E, dtype=bool)
    thresholds = np.zeros_like(E, dtype=float)

    if window <= guard:
        raise ValueError("window must be > guard so that training cells exist.")

    for i in range(n):
        # Neighborhood around CUT
        i_min = max(0, i - window)
        i_max = min(n, i + window + 1)
        neighborhood = np.arange(i_min, i_max)

        # Guard cells + CUT
        guard_min = max(i - guard, i_min)
        guard_max = min(i + guard + 1, i_max)
        guard_and_cut = np.arange(guard_min, guard_max)

        # Training cell indices
        training_idx = np.setdiff1d(neighborhood, guard_and_cut, assume_unique=True)
        N_i = training_idx.size

        if N_i == 0:
            # Fallback: no proper CFAR possible here, use global stats
            bg_sum = np.sum(E)
            N_i = n
        else:
            bg_sum = np.sum(E[training_idx])

        # Background mean (for visualization)
        E_bg[i] = bg_sum / N_i

        # CUT sum (here: M=1, just the single cell; extend if you want multi-bin CUT)
        # If you later want M>1, you'd sum/average a small window centered at i.
        cut_sum = E[i]

        # Degrees of freedom
        df1 = nu_cell * M
        df2 = nu_cell * N_i

        # F-statistic: (X/df1) / (Y/df2)  == (cut_sum * df2) / (bg_sum * df1)
        T[i] = (cut_sum * df2) / (bg_sum * df1)

        # Threshold for this N_i
        thr_i = f.ppf(1.0 - alpha, df1, df2)
        thresholds[i] = thr_i

        detections[i] = T[i] > thr_i

    return T, E_bg, detections, thresholds

def compute_cfar_rs(
    energies,
    alpha,
    window=5,
    guard=1,
):
    """
    Rank-Sum (RS) Nonparametric CFAR.

    For each CUT at index i:
        - Gather training cells from a local window excluding guard cells.
        - Compute the rank-sum statistic:
              r_i = sum( 1[E[i] >= E[j]]  for j in training cells )
        - Under H0 the rank r_i is uniform in {0,...,N_i}.
          Hence, Pfa = (N_i - T + 1)/(N_i + 1).
        - Solve for threshold T_i to achieve desired Pfa ≈ alpha.

    Parameters
    ----------
    energies : array-like
        1D energy array.
    alpha : float
        Target false alarm probability.
    window : int
        Number of cells on each side included in the local window.
    guard : int
        Number of guard cells around CUT.

    Returns
    -------
    T : ndarray
        RS test statistic (rank-sum for each CUT).
    E_bg : ndarray
        Background estimate (median over training cells).
    detections : ndarray
        Boolean detection mask.
    thresholds : ndarray
        Per-CUT threshold on rank-sum.
    """

    E = np.array(energies, dtype=float)
    n = len(E)

    T = np.zeros_like(E, dtype=float)
    E_bg = np.zeros_like(E, dtype=float)
    thresholds = np.zeros_like(E, dtype=float)
    detections = np.zeros_like(E, dtype=bool)

    if window <= guard:
        raise ValueError("window must be > guard so that training cells exist.")

    for i in range(n):
        # ---- 1. Neighborhood indices ----
        i_min = max(0, i - window)
        i_max = min(n, i + window + 1)
        neighborhood = np.arange(i_min, i_max)

        # Guard region + CUT
        guard_min = max(i - guard, i_min)
        guard_max = min(i + guard + 1, i_max)
        guard_and_cut = np.arange(guard_min, guard_max)

        # Training indices
        training_idx = np.setdiff1d(neighborhood, guard_and_cut, assume_unique=True)
        N_i = training_idx.size

        if N_i == 0:
            # Fallback: near edges.
            training_vals = E
            N_i = n
        else:
            training_vals = E[training_idx]

        # ---- 2. Background statistic for plotting ----
        E_bg[i] = np.median(training_vals)

        # ---- 3. RS statistic: count how many training cells the CUT exceeds ----
        r_i = np.sum(E[i] >= training_vals)
        T[i] = r_i

        # ---- 4. Distribution-free threshold for desired Pfa ----
        #
        # Under H0:
        #   Pfa = P(r_i >= T) = (N_i - T + 1)/(N_i + 1)
        #
        # Solve T_i from:
        #   (N_i - T_i + 1)/(N_i + 1) ≈ alpha
        #
        # Let k = floor(alpha * (N_i + 1))  → number of "top" ranks allowed
        # Then threshold is:
        #   T_i = N_i - k + 1      (1-based, but r_i is 0...N_i)
        #
        k = int(np.floor(alpha * (N_i + 1)))
        if k < 1:
            k = 1
        if k > N_i + 1:
            k = N_i + 1

        # convert to rank threshold
        T_thr = N_i - k + 1
        thresholds[i] = T_thr

        detections[i] = r_i >= T_thr

    return T, E_bg, detections, thresholds

def compute_cfar_rs_2d(
    energies2d: np.ndarray,
    alpha: float,
    window: int = 5,
    guard: int = 1,
):
    """
    2D Rank-Sum (RS) Nonparametric CFAR.

    For each CUT at (r,c):
      - Neighborhood: rows [r-window, r+window], cols [c-window, c+window]
      - Guard(+CUT): rows [r-guard, r+guard], cols [c-guard, c+guard]
      - Training: neighborhood minus guard box
      - RS statistic: r_rc = count(E[r,c] >= training_vals)
      - Under H0, r_rc is uniform in {0,...,N_rc}
        => Pfa = (N_rc - T + 1)/(N_rc + 1)
        => threshold T derived from alpha

    Returns
    -------
    T : (H,W) float
        Rank statistic per CUT.
    E_bg : (H,W) float
        Background estimate per CUT (median of training cells).
    detections : (H,W) bool
        Detection mask.
    thresholds : (H,W) float
        Rank threshold per CUT.
    """

    E = np.asarray(energies2d, dtype=float)
    H, W = E.shape

    if window <= guard:
        raise ValueError("window must be > guard so training cells exist.")

    T = np.zeros((H, W), dtype=float)
    E_bg = np.zeros((H, W), dtype=float)
    thresholds = np.zeros((H, W), dtype=float)
    detections = np.zeros((H, W), dtype=bool)

    for r in range(H):
        r0 = max(0, r - window)
        r1 = min(H, r + window + 1)
        gr0 = max(0, r - guard)
        gr1 = min(H, r + guard + 1)

        for c in range(W):
            c0 = max(0, c - window)
            c1 = min(W, c + window + 1)
            gc0 = max(0, c - guard)
            gc1 = min(W, c + guard + 1)

            # neighborhood patch
            patch = E[r0:r1, c0:c1]

            # build training mask: True everywhere except guard box (including CUT)
            train_mask = np.ones_like(patch, dtype=bool)

            # guard box coords in patch coordinates
            pr0 = gr0 - r0
            pr1 = gr1 - r0
            pc0 = gc0 - c0
            pc1 = gc1 - c0
            train_mask[pr0:pr1, pc0:pc1] = False

            training_vals = patch[train_mask]
            N = training_vals.size

            if N == 0:
                # edge fallback (rare if window>guard and not tiny images)
                training_vals = E.ravel()
                N = training_vals.size

            # background statistic
            E_bg[r, c] = np.median(training_vals)

            # RS rank statistic
            r_rc = np.sum(E[r, c] >= training_vals)
            T[r, c] = r_rc

            # threshold from alpha (same as your 1D code)
            k = int(np.floor(alpha * (N + 1)))
            k = max(1, min(k, N + 1))
            T_thr = N - k + 1
            thresholds[r, c] = T_thr

            detections[r, c] = (r_rc >= T_thr)

    return T, E_bg, detections, thresholds

def compute_cfar_os(
    energies,
    alpha,
    window=5,
    guard=1,
):
    """
    Non-parametric CFAR based on Order Statistics (OS-CFAR).

    Idea:
        For each CUT cell E[i], use training cells around it to estimate
        the local background via an order statistic. The threshold is the
        (N_i - r + 1)-th largest training sample, where r is chosen
        so that Pfa ≈ r / (N_i + 1) ≈ alpha in a distribution-free way.

        Detection rule:
            E[i] > threshold_i

        Under H0 (all cells i.i.d., continuous distribution),
        the rank of the CUT among the N_i + 1 samples (training + CUT)
        is uniform, so Pfa = r / (N_i + 1).

    Parameters
    ----------
    energies : array-like
        Input energy values (1D array).
    alpha : float
        Desired false-alarm probability Pfa.
        The actual Pfa per CUT is approximately alpha; exactly
        r / (N_i + 1) with r chosen as floor(alpha * (N_i + 1)), clamped
        to at least 1.
    window : int
        Number of cells on each side of CUT used for training+guard.
    guard : int
        Guard cells excluded on each side of CUT.

    Returns
    -------
    T : ndarray
        Test statistic per CUT (here simply T[i] = E[i]).
    E_bg : ndarray
        Background estimate per CUT (here: median of training cells).
    detections : ndarray
        Boolean detection mask.
    thresholds : ndarray
        Per-CUT threshold value (order-statistic of local training cells).
    """

    E = np.array(energies, dtype=float)
    n = len(E)

    T = np.zeros_like(E, dtype=float)
    E_bg = np.zeros_like(E, dtype=float)
    detections = np.zeros_like(E, dtype=bool)
    thresholds = np.zeros_like(E, dtype=float)

    if window <= guard:
        raise ValueError("window must be > guard so that training cells exist.")

    for i in range(n):
        # Neighborhood around CUT
        i_min = max(0, i - window)
        i_max = min(n, i + window + 1)
        neighborhood = np.arange(i_min, i_max)

        # Guard cells + CUT
        guard_min = max(i - guard, i_min)
        guard_max = min(i + guard + 1, i_max)
        guard_and_cut = np.arange(guard_min, guard_max)

        # Training cell indices: neighborhood minus guard+CUT
        training_idx = np.setdiff1d(neighborhood, guard_and_cut, assume_unique=True)
        N_i = training_idx.size

        if N_i == 0:
            # Fallback: no proper CFAR possible (near edges with tiny window)
            # Use global stats as a crude background.
            training_vals = E
            N_i = n
        else:
            training_vals = E[training_idx]

        # Background estimate for plotting: median of training cells
        E_bg[i] = np.median(training_vals)

        # ----- OS-CFAR threshold from order statistic -----
        # Sort training values (ascending)
        sorted_vals = np.sort(training_vals)

        # r = number of top values we allow CUT to exceed.
        # Under H0, Pfa = r / (N_i + 1). Choose r so that
        # Pfa ≈ alpha => r ≈ alpha * (N_i + 1).
        r = int(np.floor(alpha * (N_i + 1)))
        if r < 1:
            r = 1
        if r > N_i:
            r = N_i

        # Threshold is the (N_i - r + 1)-th order statistic (1-based),
        # i.e. index N_i - r in 0-based indexing.
        thr_val = sorted_vals[N_i - r]

        thresholds[i] = thr_val

        # Test statistic: just the CUT energy
        T[i] = E[i]

        detections[i] = T[i] > thr_val

    return T, E_bg, detections, thresholds


def compute_ca_cfar(energies, alpha, window=5, guard=1):
    E = np.array(energies)
    n = len(E)
    T = np.zeros_like(E, dtype=float)
    E_bg = np.zeros_like(E, dtype=float)
    detections = np.zeros_like(E, dtype=bool)

    N = 2 * (window - guard)  # number of training cells

    threshold_factor = N * (alpha ** (-1.0 / N) - 1)
    for i in range(n):
        i_min = max(0, i - window)
        i_max = min(n, i + window + 1)

        neighborhood = np.arange(i_min, i_max)

        # define guard cells + current cell to exclude
        guard_min = max(i - guard, i_min)
        guard_max = min(i + guard + 1, i_max)

        guard_and_cut = np.arange(guard_min, guard_max)
        training_idx = np.setdiff1d(neighborhood, guard_and_cut, assume_unique=True)

        if training_idx.size == 0:
            E_bg[i] = np.mean(E)
        else:
            E_bg[i] = np.mean(E[training_idx])

        T[i] = E[i] / E_bg[i]

        detections[i] = T[i] > threshold_factor

    return T, E_bg, detections, threshold_factor

def fft3_unshifted(video, dt):  # video: (T,H,W), dt in seconds
    T, H, W = video.shape
    C = np.fft.fftn(video, axes=(0, 1, 2))  # unshifted
    ft = np.fft.fftfreq(T, d=dt) * dt * T
    fy = np.fft.fftfreq(H, d=1.0) * H
    fx = np.fft.fftfreq(W, d=1.0) * W
    return C, ft, fy, fx

def compute_velocities_from_patch(patch,dt, bounds, best_v=False, verbose=False):
    w_patch = patch
    w_T, w_H, w_W = w_patch.shape

    F, ft, fy, fx = fft3_unshifted(w_patch, dt)
    F_shift = np.fft.fftshift(F)
    ft_shift = np.fft.fftshift(ft)
    fy_shift = np.fft.fftshift(fy)
    fx_shift = np.fft.fftshift(fx)
    
    if verbose:
        print(f"Fourier shape: {F_shift.shape}, ft range: [{ft_shift.min():.2f}, {ft_shift.max():.2f}], fy range: [{fy_shift.min():.2f}, {fy_shift.max():.2f}], fx range: [{fx_shift.min():.2f}, {fx_shift.max():.2f}]")


    energies = compute_energies_velocity_planes_torch_batched(
        velocities_px_frame=bounds,
        ft=ft_shift,
        fy=fy_shift,
        fx=fx_shift,
        F=F_shift,
        num_t=w_T,
        num_x=w_W,
        num_y=w_H,
        dc_bins=0,
        baseline_percentile=10.0,
        sigma=0.5,
        device="cuda",
    )

    alpha=1e-3

    len_bounds = bounds[2]
    window = max(1, len_bounds // 4)

    _, _, det, _ = compute_cfar_rs_2d(energies, alpha, window=window, guard=1)

    # visualize_cfar_statistics_2d(energies,det,bounds)

    velocities = np.linspace(*bounds)
    step = velocities[1] - velocities[0]
    detected_velocities = bounds[0] + np.argwhere(det) * step  # convert indices to velocity values

    if len(detected_velocities) > alpha * len(velocities)**2 + 10:

        if verbose:
            print(f"Too many detections {len(detected_velocities)}, likely blank patch. Returning zero velocity.")
        return detected_velocities , energies[det] , (0.0, 0.0)
    else:
        if verbose:
            print(f"Number of detections: {len(detected_velocities)}")

    best_energy = -np.inf
    highest_energy_velocity = (0.0, 0.0)


     # Evaluate each detected velocity


    for det_idx in range(len(detected_velocities)):
        vx = detected_velocities[det_idx][0]
        vy = detected_velocities[det_idx][1]

        # Compensate for this velocity
        F_comp = compensate_motion(F, ft, fy, fx, vx, vy, sign="minus")
        stab_image = stabilize_image_from_Fourier(F_comp, ft)

        # Compute energy of gradients of the stabilized image
        gx = np.gradient(stab_image, axis=1)
        gy = np.gradient(stab_image, axis=0)

        energy = np.sum(gx**2) + np.sum(gy**2)

        if energy > best_energy:
            best_energy = energy
            highest_energy_velocity = (vx, vy)
        if best_energy < 1e-3:
            highest_energy_velocity = (0.0, 0.0)
        
    
    if verbose:
        print(f"Detected velocity: vx={highest_energy_velocity[0]:.2f} px/frame, vy={highest_energy_velocity[1]:.2f} px/frame, gradient energy={best_energy:.2f}")


    return detected_velocities , energies[det] , highest_energy_velocity 
## Visualizations
def visualize_energies_on_single_component_velocity_planes(
    en, velocities_px_frame: tuple[float, float, int], velocities=None
):
    # velocities are given in px/frame
    if velocities is None:
        velocities = []
    v_min_px, v_max_px, num_steps = velocities_px_frame
    v_bounds_px = np.linspace(v_min_px, v_max_px, num_steps)

    # Plot vs *physical* velocity (px/frame)
    fig = plt.figure(figsize=(8, 5))
    plt.plot(v_bounds_px, en, label="Plane energy vs velocity")

    # --- DISTINCT COLORS FOR EACH VELOCITY LINE ---
    color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    for v in velocities:
        c = next(color_cycle)  # pick next color from cycle
        plt.axvline(
            x=v, linestyle="--", linewidth=2, color=c, label=f"v = {v:.2f} pixel/frame"
        )

    plt.xlabel("Velocity (pixels/frame)")
    plt.ylabel("Energy along plane")
    # plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    plt.close("all")
    return fig

def visualize_energies_on_multiple_component_velocity_planes(
   energies, bounds: tuple[float, float, int],
):
    fig=plt.figure(figsize=(6, 5))
    vel_bounds=np.linspace(*bounds)
    plt.imshow(
        energies,
        origin="lower",
        extent=[vel_bounds[0], vel_bounds[-1], vel_bounds[0], vel_bounds[-1]],
        aspect="auto",
        cmap="magma",
    )
    plt.xlabel("vy [px/frame]")
    plt.ylabel("vx [px/frame]")
    plt.colorbar(label="Energy")
    plt.tight_layout()
    plt.show()
    return fig

def visualize_cfar_statistics(
    velocities_px_frame: tuple[float, float, int],
    en: np.ndarray,
    E_bg: np.ndarray,
    T: np.ndarray,
    alpha: float,
    detections: np.ndarray = None,
    velocities=None,
):
    """
    Improved, clean, well-proportioned CFAR visualization.
    Adds red dots marking detected velocities in subplot 3.
    """

    gray = (0.25, 0.25, 0.25)

    # Prepare velocity axis
    v_min, v_max, num_steps = velocities_px_frame
    v_axis = np.linspace(v_min, v_max, num_steps)

    if velocities is None:
        velocities = []
    if detections is None:
        detections = np.zeros_like(T, dtype=bool)

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), gridspec_kw={"wspace": 0.25})

    fig.suptitle(
        "Plane Detection – Local Contrast CA-CFAR",
        fontsize=18,
        fontweight="bold",
        color=gray,
        y=1.02,
    )

    # ----------------------------------------
    # Axis styling helper
    # ----------------------------------------
    def style_axis(axis, xlabel, ylabel, title):
        axis.set_xlabel(xlabel, color=gray, fontsize=12)
        axis.set_ylabel(ylabel, color=gray, fontsize=12)
        axis.set_title(title, color=gray, fontsize=14, pad=10)

        axis.grid(True, linestyle="--", alpha=0.35)
        axis.tick_params(axis="both", colors=gray, labelsize=10)

        for side in ["left", "bottom"]:
            axis.spines[side].set_color(gray)
            axis.spines[side].set_linewidth(1.2)
        for side in ["top", "right"]:
            axis.spines[side].set_visible(False)

    # ----------------------------------------
    # 1. Plane Energy
    # ----------------------------------------
    ax = axes[0]
    ax.plot(v_axis, en, color="steelblue", linewidth=2.5)
    for v in velocities:
        ax.axvline(v, color=gray, linestyle="--", alpha=0.5)
    style_axis(ax, "Velocity $v$ (pixels/frame)", "Energy", "Plane Energy $E(v)$")

    # ----------------------------------------
    # 2. Background Energy
    # ----------------------------------------
    ax = axes[1]
    ax.plot(v_axis, E_bg, color="seagreen", linewidth=2.5)
    for v in velocities:
        ax.axvline(v, color=gray, linestyle="--", alpha=0.5)
    style_axis(
        ax,
        "Velocity $v$ (pixels/frame)",
        "Background Energy $E_{bg}(v)$",
        "Local Background",
    )

    # ----------------------------------------
    # 3. CFAR Contrast + DETECTIONS
    # ----------------------------------------
    ax = axes[2]
    ax.plot(v_axis, T, color="darkorange", linewidth=2.5)

    # Plot CFAR threshold (constant alpha)
    ax.plot(
        v_axis,
        alpha * np.ones_like(v_axis),
        color="forestgreen",
        linewidth=2,
        label="CFAR Threshold",
    )

    # 👉 DETECTION DOTS (red)
    detected_v = v_axis[detections]
    detected_T = T[detections]

    ax.scatter(detected_v, detected_T, color="red", s=35, zorder=5, label="Detections")

    for v in velocities:
        ax.axvline(v, color=gray, linestyle="--", alpha=0.5)

    style_axis(
        ax,
        "Velocity $v$ (pixels/frame)",
        "Contrast $T_{diff}$",
        "Local Contrast Statistic",
    )

    ax.legend(frameon=False, fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.close("all")
    return fig

def visualize_cfar_statistics_2d(
    energies,
    det,
    bounds: tuple[float, float, int],
    cmap="magma",
    figsize=(6, 5),
    auto_figsize=True,          # NEW: adapt figure shape to crop ratio
    base_height=4.5,            # NEW: inches (height reference when auto_figsize=True)
    marker_size: int = 60,
    marker_diam: float | None = None,
    cuts: tuple[tuple[float, float], tuple[float, float]] | None = None,
    show_legend=True,
):

    # --- torch -> numpy if needed ---
    try:
        import torch
        if isinstance(energies, torch.Tensor):
            energies = energies.detach().cpu().numpy()
        if isinstance(det, torch.Tensor):
            det = det.detach().cpu().numpy()
    except Exception:
        pass

    vmin, vmax, Nv = bounds
    v_grid = np.linspace(vmin, vmax, Nv)
    dv = float(v_grid[1] - v_grid[0]) if Nv > 1 else 1.0

    # marker sizing
    if marker_diam is not None:
        s = np.pi * (0.5 * float(marker_diam)) ** 2
    else:
        s = marker_size

    # --- cuts + clamp ---
    if cuts is None:
        vx_lo, vx_hi = vmin, vmax
        vy_lo, vy_hi = vmin, vmax
    else:
        (vx_lo, vx_hi), (vy_lo, vy_hi) = cuts

    if vx_lo > vx_hi: vx_lo, vx_hi = vx_hi, vx_lo
    if vy_lo > vy_hi: vy_lo, vy_hi = vy_hi, vy_lo

    vx_lo_c = max(vmin, vx_lo); vx_hi_c = min(vmax, vx_hi)
    vy_lo_c = max(vmin, vy_lo); vy_hi_c = min(vmax, vy_hi)

    vx_mask = (v_grid >= vx_lo_c) & (v_grid <= vx_hi_c)
    vy_mask = (v_grid >= vy_lo_c) & (v_grid <= vy_hi_c)

    if not vx_mask.any() or not vy_mask.any():
        vx_mask[:] = True
        vy_mask[:] = True
        vx_lo_c, vx_hi_c = vmin, vmax
        vy_lo_c, vy_hi_c = vmin, vmax

    energies_c = energies[np.ix_(vx_mask, vy_mask)]
    det_c      = det[np.ix_(vx_mask, vy_mask)]

    vx_vals = v_grid[vx_mask]
    vy_vals = v_grid[vy_mask]

    # extent uses pixel edges (not centers)
    extent = [
        float(vy_vals[0] - dv/2), float(vy_vals[-1] + dv/2),
        float(vx_vals[0] - dv/2), float(vx_vals[-1] + dv/2),
    ]

    # --- NEW: auto figure size based on crop ratio ---
    if auto_figsize:
        vx_range = max(1e-12, float(vx_hi_c - vx_lo_c))
        vy_range = max(1e-12, float(vy_hi_c - vy_lo_c))
        ratio = vy_range / vx_range          # width/height in data units (x=vy, y=vx)
        figsize = (base_height * ratio, base_height)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        np.log1p(energies_c),
        origin="lower",
        extent=extent,     # x=vy, y=vx
        aspect="auto",     # IMPORTANT: do not force square
        cmap=cmap,
    )

    det_idx = np.argwhere(det_c)
    if det_idx.size > 0:
        vx_det = vx_vals[det_idx[:, 0]]
        vy_det = vy_vals[det_idx[:, 1]]
        ax.scatter(
            vy_det, vx_det,
            c="red", s=s, marker="o",
            edgecolors="white", linewidths=0.8,
            label="CFAR detections",
        )

    ax.set_xlabel("vy [px/frame]")
    ax.set_ylabel("vx [px/frame]")
    fig.colorbar(im, ax=ax, label="Energy")

    if show_legend and det_idx.size > 0:
        ax.legend(loc="upper right")

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    fig.tight_layout()
    plt.show()
    return fig

def visualize_fourier_plane(
    F: np.ndarray,
    f1: np.ndarray,
    f2: np.ndarray,
    axis_to_sum: int,
    f1_min: float = None,
    f1_max: float = None,
    f2_min: float = None,
    f2_max: float = None,
    gt_velocities: list[float] = None,
    no_dc: bool = True,
    figSize=(10, 10),
    f1_label="$f_x$",
    f2_label="$f_t$",
    cmap="magma",
):
    """
    Visualize a collapsed Fourier plane from a 3D FFT cube.

    This function extracts and displays a slice of the 3D Fourier magnitude
    by summing over one axis (fx or fy), producing a 2D (ft vs fx) or
    (ft vs fy) plane. Optional cropping and overlay of theoretical
    velocity lines `f2 = -v * f1` are supported.

    Parameters
    ----------
    F : ndarray, shape (Nt, Ny, Nx)
        The full 3D FFT cube (complex-valued or magnitude).
    f1 : ndarray of float (Nx or Ny)
        Horizontal frequency axis (fx or fy).
    f2 : ndarray of float (Nt)
        Vertical frequency axis (ft).
    axis_to_sum : int
        Axis of F to collapse:
            1 → sum over fy → returns A(ft, fx)
            2 → sum over fx → returns A(ft, fy)
    f1_min, f1_max, f2_min, f2_max : float or None
        Cropping limits in physical frequency units. If any are None,
        no cropping is performed.
    gt_velocities : list of float
        Velocity values (v) for drawing lines f2 = -v f1.
    no_dc : bool
        If True, the DC component F[0,0,0] is zeroed.
    figSize : tuple
        Figure size passed to matplotlib.
    f1_label : str
        Label for horizontal axis.
    f2_label : str
        Label for vertical axis.
    cmap : str
        Matplotlib colormap name.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    """

    if gt_velocities is None:
        gt_velocities = []

    gray = (0.3, 0.3, 0.3)

    # ------------------------
    #  Remove DC if needed
    # ------------------------
    Fp = F.copy()
    if no_dc:
        Fp[0, 0, 0] = 0

    # ------------------------
    #  Sum along the chosen axis
    # ------------------------
    A = np.sum(np.abs(Fp), axis=axis_to_sum)
    A_log = np.log1p(A)

    # ------------------------
    #  Cropping logic
    # ------------------------
    if None not in (f1_min, f1_max, f2_min, f2_max):
        mask_f1 = (f1 >= f1_min) & (f1 <= f1_max)
        mask_f2 = (f2 >= f2_min) & (f2 <= f2_max)

        f1_c = f1[mask_f1]
        f2_c = f2[mask_f2]
        A_c = A_log[np.ix_(mask_f2, mask_f1)]
    else:
        f1_c = f1
        f2_c = f2
        A_c = A_log

    # ------------------------
    #  Velocity lines
    # ------------------------
    velocity_lines = []

    for v in gt_velocities:
        line = -v * f1_c

        # Mask where line is outside f2 bounds
        line_masked = np.where((line >= f2_min) & (line <= f2_max), line, np.nan)

        velocity_lines.append(line_masked)
    # ------------------------
    #  Figure setup
    # ------------------------
    fig, ax = plt.subplots(figsize=figSize)
    ax.set_aspect("equal")

    im = ax.imshow(
        A_c,
        extent=[f1_c.min(), f1_c.max(), f2_c.min(), f2_c.max()],
        origin="lower",
        cmap=cmap,
    )

    # ------------------------
    #  Axis styling (LOCAL ONLY)
    # ------------------------
    ax.set_xlabel(f1_label, color=gray)
    ax.set_ylabel(f2_label, color=gray)

    ax.tick_params(axis="x", colors=gray)
    ax.tick_params(axis="y", colors=gray)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(1.4)
        ax.spines[side].set_color(gray)

    # ------------------------
    #  Draw velocity lines
    # ------------------------
    for line in velocity_lines:
        ax.plot(f1_c, line, color=gray, linestyle="--", linewidth=1.4, alpha=0.6)

    # ------------------------
    #  Thin colorbar aligned to image (LOCAL styling)
    # ------------------------
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Power [dB]", color=gray)
    cbar.ax.tick_params(colors=gray)
    for spine in cbar.ax.spines.values():
        spine.set_color(gray)
        spine.set_linewidth(1.2)

    fig.tight_layout()
    plt.show()

    return fig

def animate_time_field(
    A,
    x_grid,
    y_grid,
    t_grid,
    colorbar_label,
    title="Field",
    video_duration=5,
    cmap="magma",
):
    """
    Animate a 3D volume A(T,Y,X) as XY frames over time t_grid.
    """
    vmin, vmax = A.min(), A.max()

    fig, ax = plt.subplots()
    im = ax.imshow(
        A[0],
        origin="lower",
        aspect="auto",
        extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    plt.colorbar(im, ax=ax, label=colorbar_label)
    txt = ax.text(
        0.02,
        0.98,
        f"t={t_grid[0]:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="w",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    def update(k):
        im.set_data(A[k])
        txt.set_text(f"t={t_grid[k]:.3f}")
        return im, txt

    fps = A.shape[0] / video_duration
    ani = FuncAnimation(fig, update, frames=A.shape[0], interval=1000 // fps, blit=True)
    plt.close(fig)
    return ani

def extract_patches_3d(X, patch_size=16, stride=8, temporal_size=-1):
    """
    Extract 3D patches from X with shape (nt, ny, nx).

    patch_size:
      - int  -> square spatial patches (py=px=patch_size)  [default behavior]
      - (py, px) tuple/list -> rectangular spatial patches

    stride:
      - int -> same stride in y and x (default)
      - (sy, sx) tuple/list -> different strides in y and x

    temporal_size:
      - -1 -> use full nt (one temporal window)
      - >0 -> window length; windows are non-overlapping (step = temporal_size),
              matching your previous behavior.
    """
    nt, ny, nx = X.shape

    # --- patch size: keep default square behavior ---
    if isinstance(patch_size, (tuple, list)):
        if len(patch_size) != 2:
            raise ValueError(f"patch_size must be int or (py, px); got {patch_size}")
        py, px = int(patch_size[0]), int(patch_size[1])
    else:
        py = px = int(patch_size)

    if py <= 0 or px <= 0:
        raise ValueError(f"patch_size must be positive; got (py,px)=({py},{px})")

    # --- stride: allow tuple, keep default int behavior ---
    if isinstance(stride, (tuple, list)):
        if len(stride) != 2:
            raise ValueError(f"stride must be int or (sy, sx); got {stride}")
        sy, sx = int(stride[0]), int(stride[1])
    else:
        sy = sx = int(stride)

    if sy <= 0 or sx <= 0:
        raise ValueError(f"stride must be positive; got (sy,sx)=({sy},{sx})")

    # --- temporal window ---
    if temporal_size == -1:
        temporal_size = nt
    temporal_size = int(temporal_size)
    if temporal_size <= 0 or temporal_size > nt:
        raise ValueError(f"temporal_size must be in [1, nt]; got {temporal_size} with nt={nt}")

    patches = []

    # Time outer loop; non-overlapping temporal blocks (step = temporal_size)
    for t in range(0, nt - temporal_size + 1, temporal_size):
        for y in range(0, ny - py + 1, sy):
            for x in range(0, nx - px + 1, sx):
                patch = X[t : t + temporal_size, y : y + py, x : x + px]
                patches.append((y, x, patch))

    return patches

def patch_window(patch_size):
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(patch_size) / (patch_size - 1))
    W = np.outer(w, w)
    return W

def window_patches(patches, patch_size=8):
    W = patch_window(patch_size)
    windowed = []
    for t, y, x, P in patches:
        windowed.append((t, y, x, P * W))
    return windowed

def show_patch_index_grid(X, patch_size=16, stride=8, patch_index=0):
    """
    X: (nt, ny, nx)
    Select patch by index, using grid generated from patch_size & stride.
    """

    # summed image
    _, ny, nx = X.shape
    img = np.sum(X, axis=0)

    # compute all grid starting coords
    coords = []
    for y in range(0, ny - patch_size + 1, stride):
        for x in range(0, nx - patch_size + 1, stride):
            coords.append((y, x))

    if patch_index < 0 or patch_index >= len(coords):
        raise ValueError(
            f"patch_index {patch_index} out of range (0..{len(coords) - 1})"
        )

    # find selected patch coordinates
    y_sel, x_sel = coords[patch_index]

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="viridis")

    # draw all patches with low alpha
    for y, x in coords:
        r = patches_l.Rectangle(
            (x, y),
            patch_size,
            patch_size,
            linewidth=0.5,
            edgecolor="white",
            facecolor="white",
            alpha=0.01,
        )
        ax.add_patch(r)

    # highlight the selected one
    r_sel = patches_l.Rectangle(
        (x_sel, y_sel),
        patch_size,
        patch_size,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        alpha=1.0,
    )
    ax.add_patch(r_sel)
    ax.set_axis_off()

    # ax.set_title(f"Patch #{patch_index} → y={y_sel}, x={x_sel}")
    plt.show()
    return fig

def apply_hann_window_3d(patch):
    """
    Applies a 3D Hann window to a spatiotemporal patch.
    
    Parameters
    ----------
    patch : np.ndarray
        Input patch with shape (Nt, Ny, Nx).
        
    Returns
    -------
    windowed_patch : np.ndarray
        The input patch multiplied by the 3D Hann window.
    """
    Nt, Ny, Nx = patch.shape
    
    # 1. Create 1D Hann windows for each dimension
    # periodic=True is generally preferred for spectral analysis (DFT)
    w_t = np.hanning(Nt)
    w_y = np.hanning(Ny)
    w_x = np.hanning(Nx)
    
    # 2. Combine into a 3D window using broadcasting
    # shape: (Nt, 1, 1) * (1, Ny, 1) * (1, 1, Nx) = (Nt, Ny, Nx)
    window_3d = w_t[:, None, None] * w_y[None, :, None] * w_x[None, None, :]
    
    # 3. Apply the window
    return patch * window_3d

def stabilize_image_from_Fourier(Cdes, ft):
    """
    Extracts the stabilized image from a desheared 3D spectrum by taking the 
    DC component (zero frequency) along the temporal axis.

    Parameters
    ----------
    Cdes : np.ndarray
        The desheared complex spectrum with shape (Nt, Ny, Nx).
    ft : np.ndarray
        The temporal frequency coordinate array.

    Returns
    -------
    img : np.ndarray
        The 2D stabilized spatial image (Ny, Nx).
    """
    dc_index = int(np.argmin(np.abs(ft)))
    
    spectrum_2d = Cdes[dc_index]
    
    return np.fft.ifft2(spectrum_2d).real

def compensate_motion(C, ft, fy, fx, vx, vy, sign="minus"):
    """
    Applies a phase ramp in the mixed domain to 'deshear' a velocity plane.
    Automatically scales input velocities to match the frequency grid dimensions.

    Parameters
    ----------
    C : np.ndarray
        Unshifted complex spectrum with shape (Nt, Ny, Nx).
    ft, fy, fx : np.ndarray
        Frequency coordinate arrays for Time, Y, and X axes.
    vx, vy : float
        Physical velocity in pixels per frame (px/frame).
    sign : str, optional
        Direction of the phase shift ('minus' or 'plus'). Defaults to 'minus'.

    Returns
    -------
    Cdes : np.ndarray
        The desheared complex spectrum.
    """
    C = np.asarray(C)
    Nt, Ny, Nx = C.shape

    range_t = np.ptp(ft)
    range_y = np.ptp(fy)
    range_x = np.ptp(fx)

    scale_x = range_t / range_x if range_x > 0 else 1.0
    scale_y = range_t / range_y if range_y > 0 else 1.0

    vx_scaled = vx * scale_x
    vy_scaled = vy * scale_y

    df = float(abs(ft[1] - ft[0]))
    dt = 1.0 / (Nt * df)
    t = np.arange(Nt) * dt

    F = np.fft.ifft(C, axis=0)

    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    dft = vx_scaled * FX + vy_scaled * FY

    sgn = 1.0 if sign == "minus" else -1.0
    phase = np.exp(1j * sgn * 2 * np.pi * dft[None, :, :] * t[:, None, None])

    return np.fft.fft(F * phase, axis=0)

def stabilize_image_from_Fourier_torch(Cdes: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
    """
    Torch/GPU version of stabilize_image_from_Fourier.

    Parameters
    ----------
    Cdes : torch.Tensor
        Complex spectrum, shape (Nt, Ny, Nx), complex64/complex128.
    ft : torch.Tensor
        Temporal frequency coords, shape (Nt,), float32/float64.

    Returns
    -------
    img : torch.Tensor
        Real stabilized image, shape (Ny, Nx), float32/float64, on same device.
    """
    if Cdes.ndim != 3:
        raise ValueError(f"Cdes must be (Nt,Ny,Nx), got {tuple(Cdes.shape)}")
    if ft.ndim != 1 or ft.shape[0] != Cdes.shape[0]:
        raise ValueError(f"ft must be (Nt,), got {tuple(ft.shape)} for Nt={Cdes.shape[0]}")

    dc_index = torch.argmin(torch.abs(ft)).item()
    spectrum_2d = Cdes[dc_index]  # (Ny, Nx)
    img = torch.fft.ifft2(spectrum_2d).real
    return img

def compensate_motion_torch(
    C: torch.Tensor,
    ft: torch.Tensor,
    fy: torch.Tensor,
    fx: torch.Tensor,
    vx: float,
    vy: float,
    sign: str = "minus",
) -> torch.Tensor:
    """
    Torch/GPU version of compensate_motion for a single (vx,vy).

    Parameters
    ----------
    C : torch.Tensor
        Unshifted complex spectrum, shape (Nt, Ny, Nx).
    ft, fy, fx : torch.Tensor
        Frequency coordinate arrays: ft (Nt,), fy (Ny,), fx (Nx,).
    vx, vy : float
        Velocity in px/frame.
    sign : str
        'minus' or 'plus'.

    Returns
    -------
    Cdes : torch.Tensor
        Desheared complex spectrum, shape (Nt, Ny, Nx), same device/dtype as C.
    """
    if C.ndim != 3:
        raise ValueError(f"C must be (Nt,Ny,Nx), got {tuple(C.shape)}")
    Nt, Ny, Nx = C.shape
    if ft.shape != (Nt,) or fy.shape != (Ny,) or fx.shape != (Nx,):
        raise ValueError(f"ft/fy/fx shapes must be ({Nt},),({Ny},),({Nx},)")

    if sign not in ("minus", "plus"):
        raise ValueError("sign must be 'minus' or 'plus'")

    # Ensure tensors are on same device
    device = C.device
    ft = ft.to(device)
    fy = fy.to(device)
    fx = fx.to(device)

    # Ranges / scaling (match your numpy logic)
    range_t = ft.max() - ft.min()
    range_y = fy.max() - fy.min()
    range_x = fx.max() - fx.min()

    scale_x = (range_t / range_x) if float(range_x) > 0 else torch.tensor(1.0, device=device, dtype=ft.dtype)
    scale_y = (range_t / range_y) if float(range_y) > 0 else torch.tensor(1.0, device=device, dtype=ft.dtype)

    vx_scaled = torch.tensor(vx, device=device, dtype=ft.dtype) * scale_x
    vy_scaled = torch.tensor(vy, device=device, dtype=ft.dtype) * scale_y

    # dt from frequency spacing (assumes uniform ft)
    df = torch.abs(ft[1] - ft[0]).to(ft.dtype)
    dt = 1.0 / (Nt * df)
    t = torch.arange(Nt, device=device, dtype=ft.dtype) * dt  # (Nt,)

    # Mixed-domain (ifft along time)
    F = torch.fft.ifft(C, dim=0)  # (Nt, Ny, Nx)

    # Build dft = vx_scaled*FX + vy_scaled*FY, with FY,FX shapes (Ny,Nx)
    FY = fy[:, None]  # (Ny,1)
    FX = fx[None, :]  # (1,Nx)
    dft = vx_scaled * FX + vy_scaled * FY  # (Ny,Nx)

    sgn = 1.0 if sign == "minus" else -1.0
    sgn = torch.tensor(sgn, device=device, dtype=ft.dtype)

    # phase shape (Nt,Ny,Nx) via broadcasting (t -> Nt,1,1; dft -> 1,Ny,Nx)
    # NOTE: This allocates a (Nt,Ny,Nx) complex tensor.
    phase_arg = sgn * (2.0 * torch.pi) * t[:, None, None] * dft[None, :, :]  # (Nt,Ny,Nx)
    phase = torch.exp(1j * phase_arg)  # complex

    # Apply ramp + FFT back along time
    Cdes = torch.fft.fft(F * phase, dim=0)
    return Cdes

def compensate_motion_torch_batched(
    C: torch.Tensor,
    ft: torch.Tensor,
    fy: torch.Tensor,
    fx: torch.Tensor,
    vxs: torch.Tensor,
    vys: torch.Tensor,
    sign: str = "minus",
    batch_size: int = 8,
) -> torch.Tensor:
    """
    Batched GPU version: compensates multiple velocities efficiently (in chunks).

    Parameters
    ----------
    C : (Nt,Ny,Nx) complex
    ft : (Nt,) float
    fy : (Ny,) float
    fx : (Nx,) float
    vxs, vys : (V,) float tensors (CPU or GPU)
    sign : 'minus'|'plus'
    batch_size : chunk size over velocities (controls memory)

    Returns
    -------
    Cdes_all : (V,Nt,Ny,Nx) complex tensor on same device as C
    """
    if sign not in ("minus", "plus"):
        raise ValueError("sign must be 'minus' or 'plus'")

    Nt, Ny, Nx = C.shape
    device = C.device
    ft = ft.to(device)
    fy = fy.to(device)
    fx = fx.to(device)
    vxs = vxs.to(device, dtype=ft.dtype)
    vys = vys.to(device, dtype=ft.dtype)

    # Scaling
    range_t = ft.max() - ft.min()
    range_y = fy.max() - fy.min()
    range_x = fx.max() - fx.min()

    scale_x = (range_t / range_x) if float(range_x) > 0 else torch.tensor(1.0, device=device, dtype=ft.dtype)
    scale_y = (range_t / range_y) if float(range_y) > 0 else torch.tensor(1.0, device=device, dtype=ft.dtype)

    # Time grid
    df = torch.abs(ft[1] - ft[0]).to(ft.dtype)
    dt = 1.0 / (Nt * df)
    t = torch.arange(Nt, device=device, dtype=ft.dtype) * dt  # (Nt,)

    # IFFT along time once
    F = torch.fft.ifft(C, dim=0)  # (Nt,Ny,Nx)

    # Spatial grids
    FY = fy[:, None]  # (Ny,1)
    FX = fx[None, :]  # (1,Nx)

    sgn = 1.0 if sign == "minus" else -1.0
    sgn = torch.tensor(sgn, device=device, dtype=ft.dtype)

    V = vxs.numel()
    out = []

    for start in range(0, V, batch_size):
        end = min(start + batch_size, V)
        vx_b = vxs[start:end] * scale_x  # (B,)
        vy_b = vys[start:end] * scale_y  # (B,)

        # dft: (B,Ny,Nx)
        dft = vx_b[:, None, None] * FX[None, :, :] + vy_b[:, None, None] * FY[None, :, :]

        # phase_arg: (B,Nt,Ny,Nx) via (B,1,Ny,Nx) * (1,Nt,1,1)
        phase_arg = sgn * (2.0 * torch.pi) * dft[:, None, :, :] * t[None, :, None, None]
        phase = torch.exp(1j * phase_arg)  # (B,Nt,Ny,Nx) complex

        # Apply: broadcast F to (B,Nt,Ny,Nx)
        Cdes_b = torch.fft.fft(F[None, :, :, :] * phase, dim=1)  # FFT along Nt axis (dim=1)
        out.append(Cdes_b)

    return torch.cat(out, dim=0)  # (V,Nt,Ny,Nx)

@torch.no_grad()
def compensate_and_stabilize_tiled_torch(
    C_t: torch.Tensor,
    ft_t: torch.Tensor,
    fy_t: torch.Tensor,
    fx_t: torch.Tensor,
    vx: float,
    vy: float,
    sign: str = "minus",
    tile_y: int = 128,
    tile_x: int = 128,
) -> torch.Tensor:
    """
    Low-memory GPU motion compensation that directly returns the stabilized image (Ny,Nx).
    Works by processing spatial tiles so we never allocate phase for the full volume.
    """
    if sign not in ("minus", "plus"):
        raise ValueError("sign must be 'minus' or 'plus'")

    Nt, Ny, Nx = C_t.shape
    device = C_t.device
    dtype_r = ft_t.dtype

    # DC index in temporal frequency
    dc_index = torch.argmin(torch.abs(ft_t)).item()

    # scaling like your original code
    range_t = ft_t.max() - ft_t.min()
    range_y = fy_t.max() - fy_t.min()
    range_x = fx_t.max() - fx_t.min()
    scale_x = (range_t / range_x) if float(range_x) > 0 else torch.tensor(1.0, device=device, dtype=dtype_r)
    scale_y = (range_t / range_y) if float(range_y) > 0 else torch.tensor(1.0, device=device, dtype=dtype_r)

    vx_scaled = torch.tensor(vx, device=device, dtype=dtype_r) * scale_x
    vy_scaled = torch.tensor(vy, device=device, dtype=dtype_r) * scale_y

    # time grid
    df = torch.abs(ft_t[1] - ft_t[0]).to(dtype_r)
    dt = 1.0 / (Nt * df)
    t = torch.arange(Nt, device=device, dtype=dtype_r) * dt  # (Nt,)

    sgn = 1.0 if sign == "minus" else -1.0
    sgn = torch.tensor(sgn, device=device, dtype=dtype_r)

    # We'll assemble ONLY the compensated DC temporal slice (Ny,Nx) in frequency domain
    spec2d = torch.empty((Ny, Nx), device=device, dtype=C_t.dtype)

    # tile loops
    for y0 in range(0, Ny, tile_y):
        y1 = min(y0 + tile_y, Ny)
        FY = fy_t[y0:y1][:, None]  # (ty,1)

        for x0 in range(0, Nx, tile_x):
            x1 = min(x0 + tile_x, Nx)
            FX = fx_t[x0:x1][None, :]  # (1,tx)

            # Take tile of spectrum (Nt,ty,tx)
            C_tile = C_t[:, y0:y1, x0:x1]

            # mixed domain along time
            F_tile = torch.fft.ifft(C_tile, dim=0)

            # per-tile dft (ty,tx)
            dft = vx_scaled * FX + vy_scaled * FY

            # phase for THIS TILE only: (Nt,ty,tx)
            phase_arg = sgn * (2.0 * torch.pi) * t[:, None, None] * dft[None, :, :]
            phase = torch.exp(1j * phase_arg).to(C_t.dtype)

            # FFT back along time and keep only DC slice
            Cdes_dc_tile = torch.fft.fft(F_tile * phase, dim=0)[dc_index]  # (ty,tx)
            spec2d[y0:y1, x0:x1] = Cdes_dc_tile

            # free tile temps
            del C_tile, F_tile, phase_arg, phase, Cdes_dc_tile

    img = torch.fft.ifft2(spec2d).real
    return img