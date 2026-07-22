"""
Utility for doing Fourier related stuff to spatio-temporal inhomogeneous
Poisson processes.
"""
import itertools
import tempfile
import time
import warnings

try:
    import cupy as cp
except ImportError:
    warnings.warn("CuPy not available. GPU acceleration will not work for quanta data. Install CuPy for GPU support.")
try:
    import cufinufft
except ImportError:
    warnings.warn("cufinufft not available. GPU acceleration will not work for continuous data. Install cufinufft for GPU support.")
import finufft
import numpy as np
from numpy.fft import fftn, fftshift, ifftn, ifftshift
from scipy.signal import get_window, windows
from scipy.stats import chi, chi2
import sparse
from tqdm import tqdm

from sptflux import datastructs, ihpp_data, utils
from sptflux.logger import get_logger


logger = get_logger(__name__)


def is_tuple_slice(x):
    return isinstance(x, tuple) and len(x) == 3


def parse_tuple_slice(x):
    if is_tuple_slice(x):
        return np.arange(*x)
    return x


def compute_amplitude_bound(npoints, V, alpha):
    """
    Returns the sqrt CFAR bound.

    This bound is slightly different from the UWB paper since we are normalizing
    the Fourier basis functions differently (by sqrt(V) instead of V).

    Args:
        npoints: number of points in the process (or weighted sum of points if windowing)
        V: volume of the process
        alpha: significance level for the CFAR bound
    """
    err = chi2.ppf(1 - alpha, 2)
    bound_val = err * npoints / (2 * V)
    # square root since we use np.abs of the coefficients, not its square
    bound_val = np.sqrt(bound_val)
    return bound_val


def center_slices(n_in, n_out, symmetric=False):
    """
    Helper for crop_fft_to_resolution

    SPATIAL AXES (centered cropping/padding) - keep central spatial frequencies
    """
    if n_out <= n_in:
        start_in = n_in // 2 - n_out // 2
        if symmetric:
            sl_in = slice(start_in, start_in + n_out + 1)
        else:
            sl_in = slice(start_in, start_in + n_out)
        sl_out = slice(None)
    else:
        start_out = n_out // 2 - n_in // 2
        sl_in = slice(None)
        if symmetric:
            sl_out = slice(start_out, start_out + n_in + 1)
        else:
            sl_out = slice(start_out, start_out + n_in)
    return sl_in, sl_out


def crop_fft_to_resolution(data_fft, target_shape):
    """
    This is not even useful since numpy's ifft has an `s` argument, which
    automatically crops/pads the input lol.

    Maybe this will be useful for soemthing else... probably not.

    Crop or pad a 3D half-spectrum FFT array (first axis contains only
    non-negative temporal frequencies as returned by rfftn(...).T)
    to match a target full-resolution shape.

    This assumes the order of the frequencies are shifted (so that DC is in
    the center for axes with both positive and negative freqs).

    Args:
        data_fft : np.ndarray or cupy.ndarray
            Complex FFT data with shape (Nt_half_in, Ny_in, Nx_in),
            where Nt_half_in == Nt_in//2 + 1 (only non-negative temporal freqs).
            This array is assumed NOT to have the time-axis fftshift applied
            (i.e. the DC component is at index 0 along axis 0).
        target_shape : tuple of ints (Nt_out, Ny_out, Nx_out)
            Desired full target shape (not half-spectrum). The returned array
            will have shape (Nt_out//2+1, Ny_out, Nx_out).

    Returns:
        cropped_fft : np.ndarray or cupy.ndarray
            Half-spectrum FFT array (first axis non-negative freqs)
            cropped/padded to correspond to the requested full resolution.
            Shape is (Nt_out//2+1, Ny_out, Nx_out).
    """
    # Input half-spectrum dims
    in_t_half, in_y, in_x = data_fft.shape
    Nt_out, Ny_out, Nx_out = target_shape

    # Output half-spectrum temporal length
    out_t_half = Nt_out // 2 + 1

    # TIME AXIS (first axis): data_fft contains only non-negative freqs.
    # Keep the lowest-frequency bins (DC and increasing positive freqs).
    n_copy_t = min(in_t_half, out_t_half)
    sl_in_t = slice(0, n_copy_t)
    sl_out_t = slice(0, n_copy_t)

    sl_in_y, sl_out_y = center_slices(in_y, Ny_out)
    sl_in_x, sl_out_x = center_slices(in_x, Nx_out)

    return data_fft[sl_in_t, sl_in_y, sl_in_x]


def pad_frames(frames, pad_t=None, pad_y=None, pad_x=None):
    """
    Pad binary frames with zeros on given sides.

    Args:
        frames: np.ndarray frames of shape (T, H, W).
        (pad_t, pad_y, pad_x): int or None
            Number of zeros to pad on each side of each axis. If None, no padding is applied.

    Returns:
        padded_frames : np.ndarray
            Padded binary frames of shape (T + 2*pad_t, H + 2*pad_y, W + 2*pad_x).
    """
    T, H, W = frames.shape
    pad_t = pad_t or 0
    pad_y = pad_y or 0
    pad_x = pad_x or 0
    padded_shape = (T + 2 * pad_t, H + 2 * pad_y, W + 2 * pad_x)
    padded_frames = np.zeros(padded_shape, dtype=frames.dtype)
    padded_frames[
        pad_t:pad_t + T,
        pad_y:pad_y + H,
        pad_x:pad_x + W
    ] = frames
    return padded_frames


def double_real_spectrum_3d(F, ft, nyquist=None):
    """
    Assumes ft is non-negative, and only the corresponding half-spectrum is given in F.

    Args:
        F: Fourier coefficients of shape (Nft, Nfy, Nfx)
        ft: either 1D array of temporal frequencies corresponding to the first axis of F,
            or a 3D grid matching the shape of F.
    """
    F_out = F.copy()
    if nyquist is None:
        mask = (ft > 0)
    else:
        mask = (ft > 0) & (ft < nyquist)
    F_out[mask] *= 2.0
    return F_out


################ 1D temporal functions ################
def drop_hermitian(ft, F=None):
    """
    Drop negative frequencies and double positive frequencies.

    Args:
        F: Fourier coefficients, shape (len(ft))
        ft: frequency axes (1D arrays)

    Returns:
        F: Fourier coefficients with Hermitian frequencies dropped (if F given)
        ft: frequency axes with Hermitian frequencies dropped
    """
    pos_fts = ft[ft >= 0]
    if F is None:
        return pos_fts
    F_new = np.copy(F)
    F_new[ft > 0] *= 2.0
    F_new = F_new[ft >= 0]
    return F_new, pos_fts


def ihpp_nufft1(
    t, weights=None, ft_max=None,
    eps=1e-6, isign=-1, two_pi=True, V=1
):
    """
    Type 1 NUFFT of points from a temporal inhomogeneous Poisson process.

    Args:
        t : array-like, shape (M,)
            Event coordinates.
        weights : array-like (M,), optional
            Complex weights (default: all ones).
        ft_max: max frequency to sweep.
        eps : requested relative tolerance for NUFFT.
        isign : +1 or -1; use -1 to match the conventional e^{-i 2π ...} FT.
        two_pi : if True, multiplies (fx, fy, ft) by 2π to convert cycles→radians.
        V: volume of events to normalize by.

    Returns:
        F : Fourier coefficients. Complex array, shape (2 * ft_max + 1)
    """
    t = np.asarray(t, dtype=np.float64)
    M = t.size

    c = np.ones(M, dtype=np.complex128) if weights is None else np.asarray(weights, dtype=np.complex128)
    c /= np.sqrt(V)

    mult = 1.0
    if two_pi:
        mult = 2 * np.pi

    N3 = int(2 * ft_max + 1)

    F = finufft.nufft1d1(t * mult, c, N3, isign=isign, eps=eps)

    return F


def ihpp_nufft3(
    t, weights=None, ft=None,
    eps=1e-6, isign=-1, two_pi=True, V=1
):
    """
    Forward NUFFT of points from a 1D temporal inhomogeneous Poisson process.

    Args:
        t : array-like, shape (M,)
            Event coordinates.
        weights : array-like (M,), optional
            Complex weights (default: all ones).
        ft : 1D arrays of frequencies (cycles per unit for two_pi=True) or
            (start, stop, step) tuple (I hardcoded it to only work for start=0).
        eps : requested relative tolerance for NUFFT.
        isign : +1 or -1; use -1 to match the conventional e^{-i 2π ...} FT.
        two_pi : if True, multiplies (fx, fy, ft) by 2π to convert cycles→radians.

    Returns:
        F : Fourier coefficients. Complex array, shape (len(ft))
        ft : the frequency axes you asked for
    """
    t = np.asarray(t, dtype=np.float64)
    M = t.size

    c = np.ones(M, dtype=np.complex128) if weights is None else np.asarray(weights, dtype=np.complex128)
    # we want the Fourier basis normalized by sqrt(volume)
    c /= np.sqrt(V)

    U = t.copy()
    if two_pi:
        U *= 2 * np.pi

    F = finufft.nufft1d3(U, c, ft, isign=isign, eps=eps)

    return F


def ihpp_inufft2(F, tt, two_pi=True, eps=1e-6, V=1, F_full=True):
    """
    Inverse NUFFT (type 2) for a 1D temporal process.
    
    Args:
        F: Fourier coefficients. Complex array of shape (len(ft))
        tt: 1D array of target coordinates
        F_full: if False, assumes only positive frequencies were given
            (and corrected already) and reconstructs
    
    Returns:
        g with shape (len(tt))
    """
    mult = 1.0
    if two_pi:
        mult = 2 * np.pi

    if not F_full:
        # if only positive frequencies were given, need to reconstruct full spectrum
        N = 2 * len(F) - 1
        F_full = np.zeros(N, dtype=np.complex128)
        F_full[len(F):] = F
        F = F_full
    # normalize basis by sqrt(volume)
    g = finufft.nufft1d2(tt * mult, F / np.sqrt(V), isign=+1, eps=eps)
    return g


def ihpp_inufft3(ft, F, tt, two_pi=True, eps=1e-6, V=1):
    """
    Inverse NUFFT (type 3) for a 1D temporal process.
    
    Args:
        ft: 1D array of frequencies
        F: Fourier coefficients. Complex array of shape (len(ft))
        tt: 1D array of target coordinates
    
    Returns:
        g with shape (len(tt))
    """
    tt = tt.copy()
    if two_pi:
        tt *= 2 * np.pi

    # normalize basis by sqrt(volume)
    g = finufft.nufft1d3(ft, F / np.sqrt(V), tt, isign=+1, eps=eps)
    return g


################ spatiotemporal functions ################
def get_3d_window(x, y, t, w=None, window="hann", cuda=False, window_t=True, window_x=True, window_y=True) -> np.ndarray:
    """
    Get separable 3D window weights evaluated at the provided coordinates.

    Window will be symmetric if the input points are inclusive [0, 1].
    For the purposes of windowing, make sure it is [0, 1) (e.g. endpoint=False).

    Args:
        x, y, t: array-like Coordinates of the points, assumed normalized to [0, 1) (e.g. endpoint=False).
        w : array-like or None
            Original weights (e.g., flux amplitudes). If None, assume 1s.
        window : str
            Window type ('hann', 'tukey', 'kaiser', etc.)
        **kwargs:
            Parameters for the window function (refer to scipy documentation).

    Returns:
        w_windowed : window weights, same shape as x, y, t
    """
    xp = cp if cuda else np
    if w is None:
        w = 1.0

    if window == "hann":
        wx = 0.5 * (1 - xp.cos(2 * xp.pi * x))
        wy = 0.5 * (1 - xp.cos(2 * xp.pi * y))
        wt = 0.5 * (1 - xp.cos(2 * xp.pi * t))
    else:
        raise ValueError(f"I did not implement continuous window type: {window}")
    
    if window_t:
        wt = xp.ones(t.shape, dtype="float64")
    if window_y:
        wy = xp.ones(y.shape, dtype="float64")
    if window_x:
        wx = xp.ones(x.shape, dtype="float64")
    # Multiply weights by separable window
    w_windowed = w * wx * wy * wt
    return w_windowed


def get_3d_window_volume(
    frames_or_shape, window="hann", cuda=False,
    window_t=True, window_x=True, window_y=True,
    **kwargs
) -> np.ndarray:
    """
    Get separable 3D window. Windows are not symmetric for the purposes of spectral analysis.

    Args:
        frames_or_shape: np.ndarray of data or shape tuple of (T, H, W)
        window (str): Window type ('hann', 'tukey', 'kaiser', etc.)
        **kwargs: Additional parameters for the window function (if applicable).
    """
    if isinstance(frames_or_shape, tuple):
        T, H, W = frames_or_shape
    else:
        T, H, W = frames_or_shape.shape
    xp = cp if cuda else np
    wt = xp.ones(T, dtype="float64")
    wy = xp.ones(H, dtype="float64")
    wx = xp.ones(W, dtype="float64")
    if window == "hann":
        if window_t:
            wt = xp.hanning(T + 1)[:-1]
        if window_y:
            wy = xp.hanning(H + 1)[:-1]
        if window_x:
            wx = xp.hanning(W + 1)[:-1]
    elif window == "gaussian":
        if "std" not in kwargs:
            stds = 1.0
        else:
            stds = kwargs["std"]
        if isinstance(stds, (int, float)):
            stds = (stds,) * 3
        std_t, std_y, std_x = stds
        if window_t:
            wt = xp.asarray(windows.gaussian(T, std=std_t, sym=False))
        if window_y:
            wy = xp.asarray(windows.gaussian(H, std=std_y, sym=False))
        if window_x:
            wx = xp.asarray(windows.gaussian(W, std=std_x, sym=False))
    elif window is None:
        pass
    else:
        raise ValueError(f"I did not implement continuous window type: {window}")

    # Create 3D window via outer products
    window_3d = wt[:, None, None] * wy[None, :, None] * wx[None, None, :]

    return window_3d


def spt_ihpp_nuft(x, y, t, fx, fy, ft, weights=None, two_pi=True):
    """
    Direct non-uniform forward Fourier transform (NUFT) via explicit sum.

    Computes F(kx, ky, f) = sum_j w_j * exp(-i * (2π?) * [kx*x_j + ky*y_j + f*t_j])

    Args:
        x, y, t : array-like, shape (M,)
            Event coordinates.
        fx, fy, ft : 1D arrays of frequencies.
        weights : array-like (M,), optional
            Complex weights (default: all ones).
        two_pi : if True, multiplies phase by 2π.

    Returns:
        F : Complex array, shape (len(fx), len(fy), len(ft))
        (fx, fy, ft) : frequency axes
    """
    x = np.asarray(x)
    y = np.asarray(y)
    t = np.asarray(t)
    M = x.size
    c = np.ones(M, dtype=np.complex128) if weights is None else np.asarray(weights, dtype=np.complex128)

    fx = np.asarray(fx)
    fy = np.asarray(fy)
    ft = np.asarray(ft)
    Nx, Ny, Nt = len(fx), len(fy), len(ft)
    F = np.zeros((Nx, Ny, Nt), dtype=np.complex128)

    pbar = tqdm(total=Nx * Ny * Nt, desc="Direct NUFT frequencies")
    for ix in range(Nx):
        for iy in range(Ny):
            for it in range(Nt):
                phase = fx[ix] * x + fy[iy] * y + ft[it] * t
                if two_pi:
                    phase = 2 * np.pi * phase
                F[ix, iy, it] = np.sum(c * np.exp(-1j * phase))
                pbar.update(1)
    pbar.close()
    return F, (fx, fy, ft)


def spt_ihpp_nufft1(
    x, y, t, weights=None,
    fx_max=None, fy_max=None, ft_max=None,
    eps=1e-6, isign=-1, two_pi=True, V=1
):
    """
    Type 1 NUFFT of points from a spatiotemporal inhomogeneous Poisson process.

    Args:
        x, y, t: array-like, shape (M,)
            Event coordinates.
        weights: array-like (M,), optional
            Complex weights (default: all ones).
        fx_max, fy_max, ft_max: (start, stop, step) tuples.
        eps: requested relative tolerance for NUFFT.
        isign: +1 or -1; use -1 to match the conventional e^{-i 2π ...} FT.
        two_pi: if True, multiplies (x, y, t) by 2π to convert cycles→radians.
        V: volume of events to normalize by.

    Returns:
        F : Fourier coefficients. Complex array, shape (len(fx), len(fy), len(ft))
        (fx, fy, ft) : the frequency axes you asked for
    """
    # check if uniform frequency grid (type 1) or arbitrary (type 3)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    M = x.size

    c = np.ones(M, dtype=np.complex128) if weights is None else np.asarray(weights, dtype=np.complex128)
    c /= np.sqrt(V)

    N1 = ft_max * 2 + 1
    N2 = fy_max * 2 + 1
    N3 = fx_max * 2 + 1

    mult = 1.0
    if two_pi:
        mult = 2 * np.pi
    # Run NUFFT
    F = finufft.nufft3d1(t * mult, y * mult, x * mult, c, (N1, N2, N3), isign=isign, eps=eps)
    return F


def spt_ihpp_nufft3(
    x, y, t, fx, fy, ft,
    weights=None, eps=1e-6, two_pi=True, V=1, cuda=False
) -> np.ndarray:
    """
    Forward type 3 NUFFT of points from a spatio-temporal inhomogeneous Poisson process.

    Args:
        x, y, t : array-like, shape (M,)
            Event coordinates.
        weights : array-like (M,), optional
            Complex weights (default: all ones).
        fx, fy, ft : 1D arrays of frequency triplets (cycles per unit for two_pi=True).
            These will not be gridded within the function.
        eps : requested relative tolerance for NUFFT.
        isign : +1 or -1; use -1 to match the conventional e^{-i 2π ...} FT.
        two_pi : if True, multiplies (fx, fy, ft) by 2π to convert cycles→radians.

    Returns:
        F : Fourier coefficients. 1D complex array, shape (len(ft))
    """
    xp = cp if cuda else np
    xnufft = cufinufft if cuda else finufft
    x = xp.asarray(x, dtype="float64")
    y = xp.asarray(y, dtype="float64")
    t = xp.asarray(t, dtype="float64")
    fx = xp.asarray(fx, dtype="float64")
    fy = xp.asarray(fy, dtype="float64")
    ft = xp.asarray(ft, dtype="float64")
    M = len(x)

    c = xp.ones(M, dtype="complex128") if weights is None else xp.asarray(weights, dtype="complex128")
    c /= xp.sqrt(V)

    mult = 1.0
    if two_pi:
        mult = 2 * xp.pi

    F = xnufft.nufft3d3(x * mult, y * mult, t * mult, c, fx, fy, ft, eps=eps, isign=-1)
    return F


def spt_ihpp_nufft3_chunked(
    x, y, t, freqgrid: datastructs.Lazy3DGrid=None, fx=None, fy=None, ft=None,
    chunk_size=10000, ft_range_max=5000, return_handler=False,
    bound=None, weights=None,
    cuda=False, **kwargs
) -> np.ndarray:
    """
    Chunked version of spt_ihpp_nufft3. Refer to that function for specific details
    about kwargs.

    Process forward NUFFT3 in chunks to avoid memory issues.

    Args:
        x, y, t: 1D arrays of event coordinates.
        freqgrid: 3D grid of frequency triplets. If given, fx, fy, ft are ignored.
        fx, fy, ft: 1D arrays of frequencies. Use these if you have a set a frequencies to evaluate
            that are not on a grid. Only used if freqgrid is None.
        bound: if given, only keep frequencies with amplitude above this bound. Otherwise, the full Fourier
            spectrum is returned.
        **kwargs: Passed to spt_ihpp_nufft3, like eps, V, two_pi, etc.

    Returns:
        tuple:
        - F: Fourier coefficients. Complex array, shape (len(ft), len(fy), len(fx)). If a bound is
            given, it is a 1D array of Fourier coefficients above the bound.
        - (ft, fy, fx): all frequency triplets above the bound. Only returned if bound is given.
    """
    xp = cp if cuda else np

    # no chunk provided, type 3 on whole data
    if chunk_size is None:
        if freqgrid is not None:
            ft, fy, fx = freqgrid[:]
            return spt_ihpp_nufft3(
                x, y, t,
                fx, fy, ft,
                weights=weights,
                cuda=cuda, **kwargs
            ).reshape(freqgrid.shape)
        elif fx is not None and fy is not None and ft is not None:
            return spt_ihpp_nufft3(
                x, y, t,
                fx, fy, ft,
                weights=weights,
                cuda=cuda, **kwargs
            )

    if freqgrid is not None:
        nfreqs = freqgrid.size
    else:
        nfreqs = len(fx)
    freq_chunk_idxs = []
    for i in tqdm(range(0, nfreqs, chunk_size), total=(nfreqs + chunk_size - 1) // chunk_size, desc="Preparing frequency chunks"):
        if freqgrid is not None:
            ft_chunk, _, _ = freqgrid[i:i+chunk_size]
        elif ft is not None:
            ft_chunk = ft[i:i+chunk_size]
        freq_chunk_idxs.extend(utils.chunk_indices(ft_chunk, chunk_size, ft_range_max, offset=i))
    n_ft_chunks = len(freq_chunk_idxs)
    n_tt_chunks = (len(t) + chunk_size - 1) // chunk_size
    pbar = tqdm(total=n_ft_chunks * n_tt_chunks, desc="Forward NUFFT ft and tt chunks")
    if bound is None:
        # find and store the full Fourier spectrum
        # will not fit into VRAM, keep as np or memmap
        tmp_memmap = utils.TempMemmap(shape=nfreqs, dtype=np.complex128)
        g = tmp_memmap.data
        for fstart, fend in freq_chunk_idxs:
            if freqgrid is not None:
                ft_chunk, fy_chunk, fx_chunk = freqgrid[fstart:fend]
            elif ft is not None and fy is not None and fx is not None:
                ft_chunk = ft[fstart:fend]
                fy_chunk = fy[fstart:fend]
                fx_chunk = fx[fstart:fend]
            ft_chunk = xp.asarray(ft_chunk, dtype="float64")
            fy_chunk = xp.asarray(fy_chunk, dtype="float64")
            fx_chunk = xp.asarray(fx_chunk, dtype="float64")
            g_chunk = xp.zeros(len(fx_chunk), dtype="complex128")
            for k in range(0, len(t), chunk_size):
                # these will be converted to xp arrays in the inner function
                weights_chunk = weights[k:k+chunk_size] if weights is not None else None
                F_part = spt_ihpp_nufft3(
                    x[k:k+chunk_size], y[k:k+chunk_size], t[k:k+chunk_size],
                    fx_chunk, fy_chunk, ft_chunk,
                    weights=weights_chunk,
                    cuda=cuda, **kwargs
                )
                g_chunk += F_part
                pbar.update(1)
            if cuda:
                g_chunk = g_chunk.get()
            g[fstart:fend] = g_chunk
            g.flush()
        pbar.close()
        if freqgrid is not None:
            g = g.reshape(freqgrid.shape)
        if return_handler:
            return g, tmp_memmap
        return g
    else:
        # only keep frequencies above the bound, toss everything else
        F_bd = []
        ft_bd = []
        fy_bd = []
        fx_bd = []
        for fstart, fend in freq_chunk_idxs:
            if freqgrid is not None:
                ft_chunk, fy_chunk, fx_chunk = freqgrid[fstart:fend]
            elif ft is not None and fy is not None and fx is not None:
                ft_chunk = ft[fstart:fend]
                fy_chunk = fy[fstart:fend]
                fx_chunk = fx[fstart:fend]
            ft_chunk = xp.asarray(ft_chunk, dtype="float64")
            fy_chunk = xp.asarray(fy_chunk, dtype="float64")
            fx_chunk = xp.asarray(fx_chunk, dtype="float64")
            g_chunk = xp.zeros(len(fx_chunk), dtype="complex128")
            for k in range(0, len(t), chunk_size):
                weights_chunk = weights[k:k+chunk_size] if weights is not None else None
                F_part = spt_ihpp_nufft3(
                    x[k:k+chunk_size], y[k:k+chunk_size], t[k:k+chunk_size],
                    fx_chunk, fy_chunk, ft_chunk,
                    weights=weights_chunk,
                    cuda=cuda, **kwargs
                )
                g_chunk += F_part
                pbar.update(1)
            mask = xp.abs(g_chunk) > bound
            # always include DC component
            # DC component will almost always be included probably
            # mask = mask | ((fx_chunk == 0) & (fy_chunk == 0) & (ft_chunk == 0))
            if xp.any(mask):
                if cuda:
                    F_bd.extend(g_chunk[mask].get())
                    ft_bd.extend(ft_chunk[mask].get())
                    fy_bd.extend(fy_chunk[mask].get())
                    fx_bd.extend(fx_chunk[mask].get())
                else:
                    F_bd.extend(g_chunk[mask])
                    ft_bd.extend(ft_chunk[mask])
                    fy_bd.extend(fy_chunk[mask])
                    fx_bd.extend(fx_chunk[mask])
        pbar.close()
        return np.array(F_bd, dtype="complex128"), (np.array(ft_bd, dtype="float64"), np.array(fy_bd, dtype="float64"), np.array(fx_bd, dtype="float64"))


def spt_ihpp_inuft(fx, fy, ft, F, xx, yy, tt, two_pi=True):
    """
    Inverse 3D NUFT via direct sum. Only shoult be used if the range of filtered
    frequencies is massive and the type 3 FFT will make a massive grid.
    
    Args:
        fx, fy, ft: x,y,z frequencies. 1D arrays of same length
        F: Fourier coefficients. Complex array of shape (len(fx))
        xx, yy, tt: (start, stop, step) tuples for target coordinates
    
    Returns:
        g with shape (len(xx), len(yy), len(tt))
    """
    fx = np.asarray(fx)
    fy = np.asarray(fy)
    ft = np.asarray(ft)
    F  = np.asarray(F)

    # Evaluate the inverse FT directly (sum over sparse frequencies)
    # fx, fy, ft are 1D arrays of frequencies (not a grid)
    # F is a 1D array of coefficients, same length as fx/fy/ft

    # Convert xx, yy, tt to arrays if needed
    xx = parse_tuple_slice(xx)
    yy = parse_tuple_slice(yy)
    tt = parse_tuple_slice(tt)

    X, Y, T = np.meshgrid(xx, yy, tt, indexing="ij")
    X = X.ravel()
    Y = Y.ravel()
    T = T.ravel()

    # Avoid outer products: sum over frequencies in a loop to save memory
    Nfreq = len(fx)
    Npts = X.size
    g_flat = np.zeros(Npts, dtype=np.complex128)
    for k in tqdm(range(Nfreq)):
        phase = fx[k] * X + fy[k] * Y + ft[k] * T
        if two_pi:
            phase = 2 * np.pi * phase
        g_flat += F[k] * np.exp(1j * phase)
    g = g_flat.reshape(len(xx), len(yy), len(tt))
    return g


def spt_ihpp_inufft3(fx, fy, ft, F, xx, yy, tt, two_pi=True, eps=1e-6, V=1, cuda=False):
    """
    Inverse NUFFT (type 3) for spatiotemporal processes.
    
    Args:
        fx, fy, ft: 1D arrays of dominant frequency triplets
        F: Fourier coefficients. Complex array of shape (len(fx))
        xx, yy, tt: 1D arrays of target coordinates
    
    Returns:
        g with shape (len(xx), len(yy), len(tt))
    """
    xp = cp if cuda else np
    xnufft = cufinufft if cuda else finufft
    fx = xp.asarray(fx, dtype="float64")
    fy = xp.asarray(fy, dtype="float64")
    ft = xp.asarray(ft, dtype="float64")
    F  = xp.asarray(F, dtype="complex128")
    xx = xp.asarray(xx, dtype="float64")
    yy = xp.asarray(yy, dtype="float64")
    tt = xp.asarray(tt, dtype="float64")

    mult = 1.0
    if two_pi:
        mult = 2 * np.pi

    g = xnufft.nufft3d3(fx, fy, ft, F / np.sqrt(V), xx * mult, yy * mult, tt * mult, isign=+1, eps=eps)
    return g


def spt_ihpp_inufft3_chunked(
    fx, fy, ft, F,
    coordgrid: datastructs.Lazy3DGrid=None, x=None, y=None, t=None,
    chunk_size=10000, ft_range_max=5000, return_handler=False, cuda=False, **kwargs
) -> np.ndarray:
    """
    Chunked version of spt_ihpp_inufft3. Refer to that function for specific details
    about kwargs.

    Chunked inverse NUFFT3 to avoid memory issues. This is chunked along time
    frequencies.

    Args:
        fx, fy, ft: 1D arrays of dominant frequencies.
        F: Fourier coefficients, shape of number of frequencies(len(fx)).
        coordgrid: 3D grid of target coordinates.
        x, y, t: 1D arrays of target coordinates, if the target cannot be represented
            as a grid. Only used if coordgrid is None.
        chunk_size: Number of points per chunk along each dimension.
        ft_range_max: Maximum range of ft values per chunk.
        **kwargs: Passed to spt_ihpp_inufft3, like eps, V, two_pi, etc.

    Returns:
        g with shape (len(xx), len(yy), len(tt)), or len(x) if coordgrid is None.
    """
    xp = cp if cuda else np

    # no chunks provided, type 3 on whole data
    if chunk_size is None and ft_range_max is None:
        if coordgrid is not None:
            tt, yy, xx = coordgrid[:]
            return spt_ihpp_inufft3(
                fx, fy, ft,
                F, xx, yy, tt,
                cuda=cuda, **kwargs
            ).reshape(coordgrid.shape)
        if x is not None and y is not None and t is not None:
            return spt_ihpp_inufft3(
                fx, fy, ft,
                F, x, y, t,
                cuda=cuda, **kwargs
            )

    if coordgrid is not None:
        ncoords = coordgrid.size
    elif x is not None and y is not None and t is not None:
        ncoords = x.size
    if ft_range_max is None:
        ft_range_max = np.inf
    if chunk_size is None:
        chunk_size = coordgrid.size
    ft_chunks = utils.chunk_indices(ft, chunk_size, ft_range_max)
    n_ft_chunks = len(ft_chunks)
    n_tt_chunks = (ncoords + chunk_size - 1) // chunk_size
    pbar = tqdm(total=n_ft_chunks * n_tt_chunks, desc="Inverse NUFFT ft and tt chunks")
    # will not fit into VRAM, keep as np or memmap
    tmp_memmap = utils.TempMemmap(shape=(ncoords), dtype=np.complex128)
    g = tmp_memmap.data
    for j in range(0, ncoords, chunk_size):
        if coordgrid is not None:
            tt_chunk, yy_chunk, xx_chunk = coordgrid[j:j+chunk_size]
        else:
            tt_chunk = t[j:j+chunk_size]
            yy_chunk = y[j:j+chunk_size]
            xx_chunk = x[j:j+chunk_size]
        tt_chunk = xp.asarray(tt_chunk, dtype="float64")
        yy_chunk = xp.asarray(yy_chunk, dtype="float64")
        xx_chunk = xp.asarray(xx_chunk, dtype="float64")
        g_chunk = xp.zeros(len(tt_chunk), dtype="complex128")
        for ft_start, ft_end in ft_chunks:
            # these will be converted to xp arrays in the inner function
            fx_chunk = fx[ft_start:ft_end]
            fy_chunk = fy[ft_start:ft_end]
            ft_chunk = ft[ft_start:ft_end]
            F_chunk = F[ft_start:ft_end]
            g_chunk += spt_ihpp_inufft3(
                fx_chunk, fy_chunk, ft_chunk,
                F_chunk,
                xx_chunk, yy_chunk, tt_chunk,
                cuda=cuda, **kwargs
            )
            pbar.update(1)
        if cuda:
            g_chunk = g_chunk.get()
        g[j:j+chunk_size] = g_chunk
        g.flush()
    pbar.close()
    if coordgrid is not None:
        g = g.reshape(coordgrid.shape)
    if return_handler:
        return g, tmp_memmap
    return g


def bound_idxs_chunked(F_memmap, bound, chunk_size=10_000_000):
    """
    Apply bound to a memmap without loading it fully in memory. Or just apply it normally in memory, idk :D.
    """
    idx_list = []
    F_shape = F_memmap.shape
    total_size = F_memmap.size

    if chunk_size is None:
        return np.nonzero(np.abs(F_memmap) > bound)
    total_iters = (total_size + chunk_size - 1) // chunk_size
    pbar = tqdm(total=total_iters, desc="Applying CFAR bound in chunks")
    for i in range(0, total_size, chunk_size):
        endidx = min(i + chunk_size, total_size)
        chunk_idxs = np.unravel_index(np.arange(i, endidx), shape=F_shape)
        # should extract as a 1D chunk
        F_chunk = F_memmap[chunk_idxs]
        mask = np.abs(F_chunk) > bound
        if np.any(mask):
            idx_local = np.nonzero(mask)[0] + i
            idx_list.append(idx_local)
        pbar.update(1)
    pbar.close()

    if len(idx_list) == 0:
        raise ValueError("No frequencies exceed the CFAR bound. Something is wrong!")

    # Concatenate all chunk indices
    keep_idx = np.concatenate(idx_list)
    return np.unravel_index(keep_idx, shape=F_shape)


def uwb3d_global_cont(
    # input events
    x, y, t,
    # frequency values to sweep
    fx, fy, ft,
    # target reconstruction coordinates
    xx=None, yy=None, tt=None,
    # grid frequencies or target coordinates
    grid_freqs=True, grid_coords=True,
    # math options
    V=None, weights=None, alpha=None, full_spectrum=False,
    herm_dropped=None, use_recon_nyquist=True, clip_flux=True,
    # performance options
    chunk_size=None, ft_range_max=None, eps=1e-6, cuda=False
):
    """
    Spatiotemporal flux probing for continuous-space points.

    Args:
        x, y, t: array of event coordinates, 1D each. SHOULD BE NORMALIZED TO [0, 1].
        fx, fy, ft: 1D arrays of coordinates for the frequency grid (these will be meshgridded if
            grid_freqs=True)
        xx, yy, tt: 1D arrays of target coordinates for the reconstruction grid (these will be
            meshgridded if grid_coords=True)
        herm_dropped: whether Hermitian equivalent frequencies were dropped (if None, will be
            inferred). In this case specifically, if all ft >=0, then herm_dropped=True.
        grid_freqs: whether the input frequencies are gridded (if True, fx, fy, ft are
            1D arrays that will be meshgridded internally). If False, fx, fy, ft are
            1D arrays of frequency triplets.
        grid_coords: whether to reconstruct on a grid (if True, x, y, t are 1D arrays that
            will be meshgridded internally). If False, x, y, t are 1D arrays of event coordinates.
        V: volume of the process
        weights: complex weights for the forward NUFFT (if None, all ones, divided by sqrt(V))
        alpha: false alarm rate for CFAR thresholding (if None, 1 / number of frequencies)
        full_spectrum: whether to return the full Fourier spectrum or only frequencies above
            the bound. This should be False if memory is an issue (too many frequencies).
        use_recon_nyquist: whether to limit the reconstruction to frequencies below the Nyquist
            frequency determined by the temporal sampling (1 / (2 * min dt)).
        clip_flux (default: True): whether to clip negative values in the reconstructed flux to zero.
        chunk_size: chunk size for NUFFT3 chunked processing
        ft_range_max: maximum number of unique time frequencies per chunk (this prevents the internal
            finufft grid from blowing up)
        eps: NUFFT epsilon tolerance
        cuda: whether to use cupy and cufinufft for GPU acceleration.

    Returns:
        tuple:
        - flux_3dfft: reconstructed flux (only returned if target coordinates are provided).
            Shape (len(tt), len(yy), len(xx)) if grid_coords=True, else shape (len(xx),)
        - F: Fourier coefficients, shape (len(ft), len(fy), len(fx)) if grid_freqs=True and full_spectrum=True.
            If grid_freqs=False and full_spectrum=True, shape (len(ft),).
            If full_spectrum=False, only a 1D array of Fourier coefficients above the bound is returned
            (regardless of grid_freqs).
        - bound: amplitude bound used
        - (ft, fy, fx): frequency triplets above bound (only returned if full_spectrum=False)
    """
    xp = cp if cuda else np
    if herm_dropped is None:
        herm_dropped = np.all(ft >= 0)
    nufftcfg_kwargs = dict(
        chunk_size=chunk_size,
        ft_range_max=ft_range_max,
        eps=eps,
        V=V,
        cuda=cuda
    )
    freqgrid = datastructs.Lazy3DGrid(ft, fy, fx)
    if grid_freqs:
        nfreqs = freqgrid.size
    else:
        nfreqs = len(ft)
    if alpha is None:
        alpha = 1 / nfreqs
    if weights is not None:
        bound = compute_amplitude_bound(np.sum(weights), V, alpha=alpha)
    else:
        bound = compute_amplitude_bound(len(x), V, alpha=alpha)
    if grid_freqs:
        logger.info(f"Probing with frequency grid of shape: {freqgrid.shape}, with {freqgrid.size} total frequencies.")
        logger.info(f"Projected size of energy grid: {freqgrid.size * 16 / (1024**3):.2f} GB")
    else:
        logger.info(f"Probing with {len(ft)} arbitrary frequency triplets.")
        logger.info(f"Projected size of energy grid: {len(ft) * 16 / (1024**3):.2f} GB")
    logger.info(f"CFAR bound: {bound}")
    x_in = xp.ascontiguousarray(xp.asarray(x, dtype="float64"))
    y_in = xp.ascontiguousarray(xp.asarray(y, dtype="float64"))
    t_in = xp.ascontiguousarray(xp.asarray(t, dtype="float64"))
    if xp.any(x_in < 0) or xp.any(x_in > 1) or xp.any(y_in < 0) or xp.any(y_in > 1) or xp.any(t_in < 0) or xp.any(t_in > 1):
        raise ValueError("Input coordinates x, y, t must be normalized to [0, 1].")
    if full_spectrum:
        if grid_freqs:
            F = spt_ihpp_nufft3_chunked(
                x_in, y_in, t_in,
                weights=weights, freqgrid=freqgrid,
                **nufftcfg_kwargs
            )
            bd_idxs = bound_idxs_chunked(F_memmap=F, bound=bound, chunk_size=chunk_size)
            F_bd = F[bd_idxs]
            ft_bd, fy_bd, fx_bd = freqgrid[bd_idxs]
        else:
            F = spt_ihpp_nufft3_chunked(
                x_in, y_in, t_in,
                weights=weights, fx=fx, fy=fy, ft=ft,
                **nufftcfg_kwargs
            )
            bd_idxs = bound_idxs_chunked(F_memmap=F, bound=bound, chunk_size=chunk_size)
            F_bd = F[bd_idxs]
            ft_bd, fy_bd, fx_bd = ft[bd_idxs], fy[bd_idxs], fx[bd_idxs]
    else:
        if grid_freqs:
            F_bd, (ft_bd, fy_bd, fx_bd) = spt_ihpp_nufft3_chunked(
                x_in, y_in, t_in,
                freqgrid=freqgrid, bound=bound,
                **nufftcfg_kwargs
            )
        else:
            F_bd, (ft_bd, fy_bd, fx_bd) = spt_ihpp_nufft3_chunked(
                x_in, y_in, t_in,
                fx=fx, fy=fy, ft=ft,
                bound=bound,
                **nufftcfg_kwargs
            )
    do_reconstruct = (xx is not None) and (yy is not None) and (tt is not None)
    if do_reconstruct:
        coordgrid = datastructs.Lazy3DGrid(tt, yy, xx)
        F_recon = F_bd.copy()
        ft_recon = ft_bd.copy()
        fy_recon = fy_bd.copy()
        fx_recon = fx_bd.copy()
        recon_nyquist = round(0.5 / np.min(np.diff(tt)))
        if herm_dropped:
            F_recon = double_real_spectrum_3d(F_recon, ft_recon, recon_nyquist)
        logger.info(f"num of allowed freqs: {len(F_bd)}")
        if use_recon_nyquist:
            nyquist_mask = (np.abs(ft_recon) <= recon_nyquist) 
            F_recon = F_recon[nyquist_mask]
            ft_recon = ft_recon[nyquist_mask]
            fy_recon = fy_recon[nyquist_mask]
            fx_recon = fx_recon[nyquist_mask]
            logger.info(f"num of allowed freqs after Nyquist limit: {len(F_bd)}")
        if grid_coords:
            out = spt_ihpp_inufft3_chunked(
                fx_recon.astype("float64"),
                fy_recon.astype("float64"),
                ft_recon.astype("float64"),
                F_recon,
                coordgrid=coordgrid,
                **nufftcfg_kwargs
            )
        else:
            out = spt_ihpp_inufft3_chunked(
                fx_recon.astype("float64"),
                fy_recon.astype("float64"),
                ft_recon.astype("float64"),
                F_recon,
                x=xx, y=yy, t=tt,
                **nufftcfg_kwargs
            )
        flux_3dfft = out.real
        if clip_flux:
            flux_3dfft = np.clip(flux_3dfft, 0, None)
        if full_spectrum:
            return flux_3dfft, F, bound
        return flux_3dfft, F_bd, bound, (ft_bd, fy_bd, fx_bd)
    if full_spectrum:
        return F, bound
    return F_bd, bound, (ft_bd, fy_bd, fx_bd)


def uwb3d_global_quanta(
    frames,
    recon_shape=None, recon_slice=None,
    V=None, alpha=None, cuda=False
):
    """
    Spatiotemporal flux probing for a volume of quanta frames.

    If you can do a global FFT, that either means it's a small video or you have a comical
    amount of VRAM/RAM/disk. I will not bother with efficiencies and just give back the whole
    Fourier spectrum.

    Returns:
        tuple:
            - F: rfftn result
            - flux
    """
    xp = cp if cuda else np
    T, H, W = frames.shape
    frames = xp.asarray(frames)
    fft_in = frames.T
    F = xp.fft.rfftn(fft_in, norm="backward").T / np.sqrt(V)
    if alpha is None:
        alpha = 1 / F.size
    bound = compute_amplitude_bound(frames.sum(), V, alpha=alpha)
    F_bd = xp.copy(F)
    F_bd[np.abs(F) < bound] = 0.0
    ifft_in = F_bd.T
    if recon_shape is not None:
        recon_shape_t, recon_shape_y, recon_shape_x = recon_shape
        out = xp.fft.irfftn(ifft_in, s=(recon_shape_x, recon_shape_y, recon_shape_t), norm="forward").T / np.sqrt(V)
    else:
        out = xp.fft.irfftn(ifft_in, norm="forward").T / np.sqrt(V)
    if recon_slice is not None:
        out = out[recon_slice]
    if cuda:
        out = out.get()
        F = F.get()
    out = np.clip(out, 0.0, None)
    return F, out


def uwb3d_stft_quanta(
    frames,
    patch_sizes=None,
    hop_sizes=None,
    window=None, window_kwargs=None,
    recon_shape=None, recon_slice=None,
    filter_func=None,
    V=None, alpha=None,
    use_recon_nyquist=False,
    save_fourier_coeffs=False, all_fourier_coeffs=False,
    cuda=False, load_frames_on_vram=False, load_output_on_vram=False
):
    """
    Spatiotemporal flux probing optimized for a volume of quanta frames using a short-time
    Fourier transform (STFT).

    This function also supports frames that contain photon counts instead of just binary arrivals.
    
    Since it's a discrete volume, a standard FFT can be used instead of the slower NUFFT.

    Args:
        frames: 3D array of shape (T, H, W) of quanta frames.
        patch_sizes: tuple of (patch_T, patch_H, patch_W). If None or 0, uses full size along that axis.
        hop_sizes: tuple of (hop_T, hop_H, hop_W). If None or 0, uses full size along that axis (no overlap).
        window: windowing function to apply per patch. If None, no windowing is applied.
        recon_shape: tuple of (Nt, Ny, Nx) for reconstruction shape. If None, no reconstruction is done.
        recon_slice: tuple of slices for (t, y, x) axes to reconstruct into. If None, reconstructs full volume.
        filter_func: func(F, ft, fy, fx) -> F_new. For any sort of filter.
        use_recon_nyquist: if True, uses the reconstruction shape to set the Fourier
            extent per chunk (only frequencies below the Nyquist frequency for the reconstruction
            will be computed). Only used if recon_shape is given.
        alpha: significance level for CFAR. Defaults to only allowing 1 false positive per patch on average.
        V: physical volume of the process (for flux to be in physical units).
        all_fourier_coeffs: USE WITH CAUTION. If True, saves all fourier coefficients
            for each chunk, otherwise only saves ones above CFAR bound as a sparse
            array.
        load_frames_on_vram (bool): Only checked if cuda=True. A full 100kx512x512 quanta
            sequence can fit in around ~26 GB of VRAM, so if your patch sizes are small
            enough, you can load the quanta frames directly onto VRAM for a significant(?)
            performance boost.
        load_output_on_vram (bool): Only checked if cuda=True. If True, loads the output
            of each patch onto VRAM for faster(?) reconstruction.
    
    Returns:
        tuple:
            - F: only returned if save_fourier_coeffs is True. Half-spectrum Fourier
                coefficients (for ft >= 0, uncorrected for 2x). Either a dense 6D
                array coefficients per patch (if all_fourier_coeffs=True), or a
                sparse array of coefficients above the CFAR bound.
            - flux: only returned if Nt, Ny, Nx are given. Reconstructed flux volume
                of shape (Nt, Ny, Nx).
    """
    try:
        xp = cp if cuda else np
    except Exception as e:
        raise ImportError(f"CuPy not available even though cuda=True. Install CuPy for GPU acceleration, or set cuda=False.") from e
    T, H, W = frames.shape
    # load it into memory, it won't fit in VRAM along with FFT intermediates
    if np.all(frames < 256):
        frames_dtype = "uint8"
    elif np.all(frames < 65536):
        frames_dtype = "uint16"
    else:
        frames_dtype = "uint32"
    frames = np.asarray(frames, dtype=frames_dtype)

    do_reconstruct = recon_shape is not None
    if not do_reconstruct and use_recon_nyquist:
        raise ValueError("Cannot use use_recon_nyquist if not doing a reconstruction.")
    # precompute chunk grid ranges (only compute patches completely inside volume for now)
    patch_T, patch_H, patch_W = patch_sizes
    hop_T, hop_H, hop_W = hop_sizes

    window_t = True
    window_y = True
    window_x = True
    window_kwargs = window_kwargs if window_kwargs is not None else {}
    # deal with unwindowed dimensions or edge case windows. a dimension that
    # isn't windowed is also not padded (see pad step below), so its hop/patch
    # values are set to stay consistent with the unpadded extent.
    if patch_T in (None, 0) or hop_T in (None, 0):
        hop_T = 0
        patch_T = T
        window_t = False
    elif patch_T == 1:
        hop_T = 1
        window_t = False
    if patch_H in (None, 0) or hop_H in (None, 0):
        hop_H = 0
        patch_H = H
        window_y = False
    elif patch_H == 1:
        hop_H = 1
        window_y = False
    if patch_W in (None, 0) or hop_W in (None, 0):
        hop_W = 0
        patch_W = W
        window_x = False
    elif patch_W == 1:
        hop_W = 1
        window_x = False

    # padding data to avoid edge artifacts; only pad windowed dimensions
    frames_pad = pad_frames(
        frames,
        pad_t=patch_T if window_t else 0,
        pad_y=patch_H if window_y else 0,
        pad_x=patch_W if window_x else 0,
    )
    T_pad, H_pad, W_pad = frames_pad.shape
    if cuda and load_frames_on_vram:
        frames_pad = cp.asarray(frames_pad)

    # chunk start positions per dimension (consistent with the padding above)
    if hop_T == 0:  # single full-dimension patch
        t_starts = [0]
    elif not window_t:  # patch_T == 1, one patch per (unpadded) slice
        t_starts = list(range(0, T_pad))
    else:
        t_starts = list(range(0, T_pad - patch_T + 1, hop_T))
    if hop_H == 0:
        h_starts = [0]
    elif not window_y:
        h_starts = list(range(0, H_pad))
    else:
        h_starts = list(range(0, H_pad - patch_H + 1, hop_H))
    if hop_W == 0:
        w_starts = [0]
    elif not window_x:
        w_starts = list(range(0, W_pad))
    else:
        w_starts = list(range(0, W_pad - patch_W + 1, hop_W))
    nchunks_t = len(t_starts)
    nchunks_h = len(h_starts)
    nchunks_w = len(w_starts)

    if do_reconstruct:
        # these will not fit into GPU VRAM, keep them as numpy arrays
        # keep these as float32 to save memory
        # we don't need that much precision anyway
        Nt, Ny, Nx = recon_shape
        # only pad reconstruction axes that were windowed/padded on the input,
        # so padded axes stay consistent with frames_pad (unwindowed axes get 0)
        recon_pad = (
            int(Nt * patch_T / T) if window_t else 0,
            int(Ny * patch_H / H) if window_y else 0,
            int(Nx * patch_W / W) if window_x else 0,
        )
        recon_shape_pad = (Nt + 2 * recon_pad[0], Ny + 2 * recon_pad[1], Nx + 2 * recon_pad[2])
        recon_slice_inner = (slice(recon_pad[0], recon_pad[0] + Nt), slice(recon_pad[1], recon_pad[1] + Ny), slice(recon_pad[2], recon_pad[2] + Nx))
        Nt_pad, Ny_pad, Nx_pad = recon_shape_pad
        if recon_slice is not None:
            recon_t_slice, recon_y_slice, recon_x_slice = recon_slice
            # lol
            if recon_t_slice.start is None:
                recon_t_slice = slice(0, recon_t_slice.stop, recon_t_slice.step)
            if recon_y_slice.start is None:
                recon_y_slice = slice(0, recon_y_slice.stop, recon_y_slice.step)
            if recon_x_slice.start is None:
                recon_x_slice = slice(0, recon_x_slice.stop, recon_x_slice.step)
            Nt_new = len(range(*recon_t_slice.indices(Nt)))
            Ny_new = len(range(*recon_y_slice.indices(Ny)))
            Nx_new = len(range(*recon_x_slice.indices(Nx)))
            out = np.zeros((Nt_new, Ny_new, Nx_new), dtype="float32")
        else:
            out = np.zeros((Nt, Ny, Nx), dtype="float32")
        wsum = np.zeros_like(out, dtype="float32")
        if cuda and load_output_on_vram:
            out = cp.asarray(out)
            wsum = cp.asarray(wsum)
        logger.info(f"Reconstruction shape {out.shape}, padded shape {recon_shape_pad}, recon slice {recon_slice}, inner recon slice {recon_slice_inner}")

    total_chunks = len(t_starts) * len(h_starts) * len(w_starts)

    # volumes and freq extents for each chunk
    V_chunk = (patch_T / T) * (patch_H / H) * (patch_W / W) * V
    nfx = patch_W
    nfy = patch_H
    nft = patch_T
    nft_half = nft // 2 + 1
    if do_reconstruct:
        # estimated output sizes for this chunk mapping into global Nx,Ny,Nt
        out_t_size_chunk = int(Nt * patch_T / T)
        out_y_size_chunk = int(Ny * patch_H / H)
        out_x_size_chunk = int(Nx * patch_W / W)
        logger.info(f"Chunk Fourier size: (nft, nfy, nfx) = ({nft}, {nfy}, {nfx}), output size per chunk: ({out_t_size_chunk}, {out_y_size_chunk}, {out_x_size_chunk})")
        if use_recon_nyquist:
            nfx = out_x_size_chunk
            nfy = out_y_size_chunk
            nft = out_t_size_chunk
            nft_half = nft // 2 + 1
        ft = xp.fft.rfftfreq(patch_T, 1 / patch_T)
        fy = xp.fft.fftfreq(patch_H, 1 / patch_H)
        fx = xp.fft.fftfreq(patch_W, 1 / patch_W)
        ftt, fyy, fxx = xp.meshgrid(ft, fy, fx, indexing="ij")
        weight_output = get_3d_window_volume(
            (out_t_size_chunk, out_y_size_chunk, out_x_size_chunk), window=window, cuda=cuda,
            window_t=window_t, window_x=window_x, window_y=window_y,
            **window_kwargs
        )
        weight_output = weight_output
        weight_output_sqrt = xp.sqrt(weight_output)
        weight_output_np = weight_output.get() if cuda else weight_output
    if save_fourier_coeffs:
        if all_fourier_coeffs:
            # 6d array of (t_chunk, h_chunk, w_chunk, ft, fy, fx)
            bytesize = nchunks_t * nchunks_h * nchunks_w * nfx * nfy * nft_half * 16
            gbsize = bytesize / (1024 ** 3)
            logger.info(f"Allocating full Fourier blocks of size {gbsize:.2f} GB")
            if gbsize > 4.0:
                logger.warning(f"Fourier blocks size is very large ({gbsize:.2f} GB). Are you SURE you want to do this? Consider setting all_fourier_coeffs=False to save only coefficients above CFAR bound.")
            fourier_blocks = np.zeros(
                (nchunks_t, nchunks_h, nchunks_w, nft_half, nfy, nfx), dtype="complex128"
            )
        else:
            # save fourier values as sparse, and only ones above CFAR
            fourier_coords = [[] for _ in range(6)]
            fourier_vals = []
            fourier_shape = (nchunks_t, nchunks_h, nchunks_w, nft_half, nfy, nfx)

    if alpha is None:
        # on average, only allow 1 false positive per chunk by default
        alpha = 1 / (nft_half * nfy * nfx)
    # keep as float64 since fft promotes to float64 anyway
    weight_input = get_3d_window_volume(
        (patch_T, patch_H, patch_W), window=window, cuda=cuda,
        window_t=window_t, window_x=window_x, window_y=window_y,
        **window_kwargs
    )
    weight_input_sqrt = xp.sqrt(weight_input)

    pbar = tqdm(total=total_chunks, desc="3D STFT reconstruction")

    for i, j, k in itertools.product(
        range(len(t_starts)), range(len(h_starts)), range(len(w_starts))
    ):
        t0, h0, w0 = t_starts[i], h_starts[j], w_starts[k]

        # skip chunks that don't intersect with recon slice (if given)
        # put this early so we can skip all processing
        if do_reconstruct:
            # target output grid sizes for this chunk mapping into global Nx,Ny,Nt
            tsl_start = int(Nt_pad * t0 / T_pad)
            hsl_start = int(Ny_pad * h0 / H_pad)
            wsl_start = int(Nx_pad * w0 / W_pad)
            tsl_out = slice(tsl_start, tsl_start + out_t_size_chunk)
            hsl_out = slice(hsl_start, hsl_start + out_y_size_chunk)
            wsl_out = slice(wsl_start, wsl_start + out_x_size_chunk)

            # this is what padding made me do
            # help me
            # padded inner region bounds
            t_inner = recon_slice_inner[0]  # slice(recon_pad[0], recon_pad[0]+Nt)
            y_inner = recon_slice_inner[1]
            x_inner = recon_slice_inner[2]

            if recon_slice is not None:
                # normalize user recon slices to unpadded Nt/Ny/Nx BEFORE shifting
                rt0, rt1, rts = recon_t_slice.indices(Nt)
                ry0, ry1, rys = recon_y_slice.indices(Ny)
                rx0, rx1, rxs = recon_x_slice.indices(Nx)

                # selection in padded coords
                t_sel_pad = slice(rt0 + recon_pad[0], rt1 + recon_pad[0], rts)
                y_sel_pad = slice(ry0 + recon_pad[1], ry1 + recon_pad[1], rys)
                x_sel_pad = slice(rx0 + recon_pad[2], rx1 + recon_pad[2], rxs)

                # sel_origin in padded coords (defines compact indexing)
                t_origin = rt0 + recon_pad[0]
                y_origin = ry0 + recon_pad[1]
                x_origin = rx0 + recon_pad[2]
            else:
                # full inner region selection (no downsample)
                t_sel_pad = t_inner
                y_sel_pad = y_inner
                x_sel_pad = x_inner
                t_origin = t_inner.start
                y_origin = y_inner.start
                x_origin = x_inner.start
            # n_global are Nt_pad/Ny_pad/Nx_pad
            t_out_slc, t_chunk_slc = utils.map_chunk_to_compact_selection(
                chunk_start=tsl_out.start, chunk_stop=tsl_out.stop,
                inner_start=t_inner.start, inner_stop=t_inner.stop,
                sel=t_sel_pad, n_global=Nt_pad, sel_origin=t_origin
            )
            y_out_slc, y_chunk_slc = utils.map_chunk_to_compact_selection(
                chunk_start=hsl_out.start, chunk_stop=hsl_out.stop,
                inner_start=y_inner.start, inner_stop=y_inner.stop,
                sel=y_sel_pad, n_global=Ny_pad, sel_origin=y_origin
            )
            x_out_slc, x_chunk_slc = utils.map_chunk_to_compact_selection(
                chunk_start=wsl_out.start, chunk_stop=wsl_out.stop,
                inner_start=x_inner.start, inner_stop=x_inner.stop,
                sel=x_sel_pad, n_global=Nx_pad, sel_origin=x_origin
            )

            all_slice_reconsl_out = (t_out_slc, y_out_slc, x_out_slc)
            all_slice_intersect_patch = (t_chunk_slc, y_chunk_slc, x_chunk_slc)

            if out[all_slice_reconsl_out].size == 0:
                pbar.update(1)
                continue

        # extract raw binned chunk (should be contained within volume, so full patch size)
        spadchunk = frames_pad[t0:t0+patch_T, h0:h0+patch_H, w0:w0+patch_W]
        spadchunk = xp.asarray(spadchunk, dtype=frames_dtype)

        if spadchunk.shape != (patch_T, patch_H, patch_W):
            raise ValueError(f"Chunk shape {spadchunk.shape} mismatch with patch size {(patch_T, patch_H, patch_W)}, something is wrong.")

        n_events_weighted = xp.sum(spadchunk * weight_input)
        # n_events = xp.count_nonzero(spadchunk)
        if n_events_weighted == 0:
            pbar.update(1)
            continue
        # windowed amplitude bound
        bound_chunk = compute_amplitude_bound(
            n_events_weighted, V_chunk, alpha=alpha
        )

        # perform as regular FFT on binned patch
        # transpose so T is last axis, so we only have positive ft
        # transpose again for expected (ft, fy, fx) ordering
        fftin = (spadchunk * weight_input_sqrt).T
        # fftin = xp.transpose(spadchunk * weight_input, (2, 1, 0))
        F_chunk = xp.fft.rfftn(
            fftin, axes=(0, 1, 2), norm="backward"
        ) / xp.sqrt(V_chunk)
        F_chunk = F_chunk.T
        if filter_func is not None:
            F_chunk = filter_func(F_chunk, ftt, fyy, fxx)
        F_chunk_abs = xp.abs(F_chunk)

        if save_fourier_coeffs:
            # unshifted, take only positive ft
            # shift the spatial dimensions though to keep consistent
            # with fiNUFFT ordering
            # and when t_chunk_bins is even, it takes negative nyquist
            # coefficient with negative nyquist should be real, don't have
            # to take conjugate
            if use_recon_nyquist:
                F_chunk_tosave = crop_fft_to_resolution(
                    xp.fft.fftshift(F_chunk, axes=(1, 2)),
                    (out_t_size_chunk, out_y_size_chunk, out_x_size_chunk),
                )
            else:
                F_chunk_tosave = xp.fft.fftshift(F_chunk, axes=(1, 2))
            if all_fourier_coeffs:
                fourier_blocks[i, j, k, :, :, :] = F_chunk_tosave
            else:
                # where above cfar
                cfarwhere = xp.nonzero(xp.abs(F_chunk_tosave) > bound_chunk)
                bound_vals = F_chunk_tosave[cfarwhere]
                chunkidx_t = xp.full_like(cfarwhere[0], fill_value=i)
                chunkidx_h = xp.full_like(cfarwhere[1], fill_value=j)
                chunkidx_w = xp.full_like(cfarwhere[2], fill_value=k)
                # append to list is way faster than doing xp.concatenate every time, I think?
                if cuda:
                    # keep these off GPU to save VRAM
                    chunkidx_t = chunkidx_t.get()
                    chunkidx_h = chunkidx_h.get()
                    chunkidx_w = chunkidx_w.get()
                    cfarwhere = (cfarwhere[0].get(), cfarwhere[1].get(), cfarwhere[2].get())
                    bound_vals = bound_vals.get()
                fourier_coords[0].append(chunkidx_t)
                fourier_coords[1].append(chunkidx_h)
                fourier_coords[2].append(chunkidx_w)
                fourier_coords[3].append(cfarwhere[0])
                fourier_coords[4].append(cfarwhere[1])
                fourier_coords[5].append(cfarwhere[2])
                fourier_vals.append(bound_vals)

        if do_reconstruct:
            # CFAR thresholding
            # we're not using F_chunk for anything else anymore, just modify
            # it directly
            F_chunk_bd = F_chunk
            # only mask positive ft, just set negative ft to zero
            # should be half the computations probably?
            # mask of all below cfar
            bd_mask = F_chunk_abs < bound_chunk
            # basic slice should return view
            F_chunk_bd[bd_mask] = 0.0

            # inverse NUFFT to chunk grid
            # norm forward to disable normalization
            # shape reversed due to transpose
            ifftin = F_chunk_bd.T
            out_chunk = xp.fft.irfftn(
                ifftin, s=(out_x_size_chunk, out_y_size_chunk, out_t_size_chunk), axes=(0, 1, 2), norm="forward"
            ) / xp.sqrt(V_chunk)
            out_chunk = out_chunk.T

            # clip at the end instead?
            # flux_chunk = xp.clip(out_chunk.real, a_min=0.0, a_max=None)
            flux_chunk = out_chunk.real

            # weighted chunk and accumulate into output and weight-sum
            weighted_chunk = flux_chunk * weight_output_sqrt
            # weighted_chunk = flux_chunk * weight_output
            # weighted_chunk = flux_chunk

            # compute output slice indices into final Nx,Ny,Nt grid
            if cuda and load_output_on_vram:
                try:
                    out[all_slice_reconsl_out] += weighted_chunk[all_slice_intersect_patch]
                except ValueError as e:
                    print(f"Error updating output: {e}")
                    print(f"weighted_chunk shape: {weighted_chunk.shape}")
                    print(f"weighted_chunk slice shape: {weighted_chunk[all_slice_intersect_patch].shape}")
                    print(f"all_slice_intersect_patch: {all_slice_intersect_patch}")
                    print(f"out shape: {out.shape}")
                    print(f"out slice shape: {out[all_slice_reconsl_out].shape}")
                    print(f"all_slice_reconsl_out: {all_slice_reconsl_out}")
                    raise
                wsum[all_slice_reconsl_out] += weight_output[all_slice_intersect_patch]
            else:
                if cuda:
                    weighted_chunk = weighted_chunk.get()
                out[all_slice_reconsl_out] += weighted_chunk[all_slice_intersect_patch]
                wsum[all_slice_reconsl_out] += weight_output_np[all_slice_intersect_patch]

        pbar.update(1)
    pbar.close()

    # no longer needed, free up VRAM in case it isn't automatically freed
    del frames_pad

    if save_fourier_coeffs:
        if all_fourier_coeffs:
            F = fourier_blocks
        else:
            # coordinates and vals not on GPU
            fourier_coords = np.array([np.concatenate(dim_coords) for dim_coords in fourier_coords])
        fourier_vals = np.concatenate(fourier_vals)
        F = sparse.COO(fourier_coords, fourier_vals, shape=fourier_shape)

    if do_reconstruct:
        if cuda and load_output_on_vram:
            out = out.get()
            wsum = wsum.get()
        # normalize by accumulated window weights (avoid divide-by-zero)
        flux = np.clip(out, a_min=0.0, a_max=None) / (np.asarray(wsum, dtype="float32") + 1e-6)
        if save_fourier_coeffs:
            return F, flux
        return flux
    return F


def uwb3d_stft_quanta_speed(
    frames,
    patch_sizes=None,
    hop_sizes=None,
    window=None, window_kwargs=None,
    recon_shape=None,
    V=None, alpha=None
):
    """
    Multithreaded GPU version for SPEEEEEDDD.
    """
    T, H, W = frames.shape
    # load it into memory, it won't fit in VRAM along with FFT intermediates
    frames = np.asarray(frames, dtype="uint8")

    # precompute chunk grid ranges (only compute patches completely inside volume for now)
    patch_T, patch_H, patch_W = patch_sizes
    hop_T, hop_H, hop_W = hop_sizes
    window_t = True
    window_y = True
    window_x = True
    # deal with unwindowed dimensions or edge case windows
    if patch_T in (None, 0) or hop_T in (None, 0):
        t_starts = [0]
        hop_T = 0
        patch_T = T
        window_t = False
    elif patch_T == 1:
        t_starts = list(range(0, T))
        hop_T = 1
        window_t = False
    else:
        t_starts = list(range(0, T - patch_T + 1, hop_T))
    if patch_H in (None, 0) or hop_H in (None, 0):
        h_starts = [0]
        hop_H = 0
        patch_H = H
        window_y = False
    elif patch_H == 1:
        h_starts = list(range(0, H))
        hop_H = 1
        window_y = False
    else:
        h_starts = list(range(0, H - patch_H + 1, hop_H))
    if patch_W in (None, 0) or hop_W in (None, 0):
        w_starts = [0]
        hop_W = 0
        patch_W = W
        window_x = False
    elif patch_W == 1:
        w_starts = list(range(0, W))
        hop_W = 1
        window_x = False
    else:
        w_starts = list(range(0, W - patch_W + 1, hop_W))

    Nt, Ny, Nx = recon_shape
    # these will not fit into GPU VRAM, keep them as numpy arrays
    out = np.zeros((Nt, Ny, Nx), dtype="float64")
    wsum = np.zeros_like(out)

    total_chunks = len(t_starts) * len(h_starts) * len(w_starts)

    # volumes and freq extents for each chunk
    V_chunk = (patch_T / T) * (patch_H / H) * (patch_W / W) * V
    V_chunk_sqrt = np.sqrt(V_chunk)
    nfx = patch_W
    nfy = patch_H
    nft = patch_T
    nft_half = nft // 2 + 1
    # estimated output sizes for this chunk mapping into global Nx,Ny,Nt
    out_t_size_chunk = int(Nt * patch_T / T)
    out_y_size_chunk = int(Ny * patch_H / H)
    out_x_size_chunk = int(Nx * patch_W / W)
    nfx = out_x_size_chunk
    nfy = out_y_size_chunk
    nft = out_t_size_chunk
    nft_half = nft // 2 + 1

    input_size = (patch_T, patch_H, patch_W)
    input_size_T = (patch_W, patch_H, patch_T)
    output_size = (out_t_size_chunk, out_y_size_chunk, out_x_size_chunk)
    output_size_T = (out_x_size_chunk, out_y_size_chunk, out_t_size_chunk)
    window_kwargs = window_kwargs if window_kwargs is not None else {}
    weight_input = get_3d_window_volume(
        input_size, window=window, cuda=False,
        window_t=window_t, window_x=window_x, window_y=window_y,
        **window_kwargs
    )
    weight_input_sqrt = np.sqrt(weight_input)
    weight_output = get_3d_window_volume(
        output_size, window=window, cuda=False,
        window_t=window_t, window_x=window_x, window_y=window_y,
        **window_kwargs
    )
    weight_output_sqrt = np.sqrt(weight_output)

    if alpha is None:
        alpha = 1 / (nft_half * nfy * nfx)

    # CUDA path: distribute chunks across available GPUs and run concurrently
    import concurrent.futures
    try:
        n_gpus = cp.cuda.runtime.getDeviceCount()
    except Exception:
        n_gpus = 1
    if n_gpus <= 0:
        n_gpus = 1
    
    # warm up GPUs for FFTs (yes, seriously)
    # for dev in range(n_gpus):
    #     with cp.cuda.Device(dev):
    #         # force context and pool init
    #         cp.zeros((1,))
    #         # warmup FFT plans
    #         dummy = cp.zeros(input_size, dtype="float64")
    #         cp.fft.rfftn(dummy.T, axes=(0, 1, 2), norm="backward")
    #         dummy = cp.zeros(output_size, dtype="complex128")
    #         cp.fft.irfftn(dummy.T, s=output_size_T, axes=(0, 1, 2), norm="forward")

    # build list of chunk indices
    all_chunks = list(itertools.product(
        range(len(t_starts)), range(len(h_starts)), range(len(w_starts))
    ))
    # partition round-robin across GPUs
    per_device_chunks = [[] for _ in range(n_gpus)]
    for idx, chunk in enumerate(all_chunks):
        per_device_chunks[idx % n_gpus].append(chunk)

    # worker: runs on a single GPU and returns a list of updates to apply on host
    def _worker(device_id, assigned_chunks):
        with cp.cuda.Device(device_id):
            # warm up GPUs for FFTs (yes, seriously)
            # force context and pool init
            cp.zeros((1,))
            # warmup FFT plans
            dummy = cp.zeros(input_size, dtype="float64").T
            # print("dummy1", dummy.shape, dummy.dtype, dummy.flags)
            cp.fft.rfftn(dummy, axes=(0, 1, 2), norm="backward")
            dummy = cp.zeros((nfx, nfy, nft_half), dtype="complex128").T
            # print("dummy2", dummy.shape, dummy.dtype, dummy.flags)
            cp.fft.irfftn(dummy, s=output_size_T, axes=(0, 1, 2), norm="forward")
            cp.cuda.Stream.null.synchronize()  # ensure warmup completes

            # local_updates = []  # list of (tsl_out, hsl_out, wsl_out, weighted_chunk_np, weight_output_np)
            local_out = cp.zeros((Nt, Ny, Nx), dtype="float64")
            local_wsum = cp.zeros_like(local_out)
            # per-worker recomputation on this device
            local_weight_input = cp.asarray(weight_input)
            local_weight_input_sqrt = cp.asarray(weight_input_sqrt)
            local_weight_output = cp.asarray(weight_output)
            local_weight_output_sqrt = cp.asarray(weight_output_sqrt)
            time_per_chunk = []

            for (i, j, k) in assigned_chunks:
                runtime_start = time.time()
                t0, h0, w0 = t_starts[i], h_starts[j], w_starts[k]
                spadchunk = frames[t0:t0+patch_T, h0:h0+patch_H, w0:w0+patch_W]
                spadchunk = cp.asarray(spadchunk, dtype="uint8")

                n_events_weighted = cp.sum(spadchunk * local_weight_input)
                # n_events = xp.count_nonzero(spadchunk)
                if n_events_weighted == 0:
                    pbar.update(1)
                    continue

                bound_chunk = compute_amplitude_bound(
                    n_events_weighted, V_chunk, alpha=alpha
                )

                fftin = (spadchunk * local_weight_input_sqrt).T
                # print("fftin", fftin.shape, fftin.dtype, fftin.flags)
                F_chunk = cp.fft.rfftn(
                    fftin, axes=(0, 1, 2), norm="backward"
                )
                F_chunk = F_chunk.T

                tsl_out = slice(int(Nt * t0 / T), int(Nt * (t0 + patch_T) / T))
                hsl_out = slice(int(Ny * h0 / H), int(Ny * (h0 + patch_H) / H))
                wsl_out = slice(int(Nx * w0 / W), int(Nx * (w0 + patch_W) / W))

                F_chunk_bd = crop_fft_to_resolution(cp.fft.fftshift(F_chunk, axes=(1, 2)), output_size) / V_chunk_sqrt
                bd_mask = cp.abs(F_chunk_bd) < bound_chunk
                F_chunk_bd[bd_mask] = 0.0

                ifftin = cp.fft.ifftshift(F_chunk_bd, axes=(1, 2)).T
                # print("ifftin", ifftin.shape, ifftin.dtype, ifftin.flags)
                out_chunk = cp.fft.irfftn(
                    ifftin, s=output_size_T, axes=(0, 1, 2), norm="forward"
                ) / V_chunk_sqrt
                out_chunk = out_chunk.T

                flux_chunk = cp.clip(out_chunk.real, a_min=0.0, a_max=None)
                weighted_chunk = flux_chunk * local_weight_output_sqrt

                # move to host and store update
                # weighted_chunk_np = weighted_chunk.get()
                # weight_output_np = weight_output.get()
                # local_updates.append((tsl_out, hsl_out, wsl_out, weighted_chunk_np, weight_output_np))
                local_out[tsl_out, hsl_out, wsl_out] += weighted_chunk
                local_wsum[tsl_out, hsl_out, wsl_out] += local_weight_output
                runtime_end = time.time()
                time_per_chunk.append(runtime_end - runtime_start)
        avg_time = np.mean(time_per_chunk)
        total_time = np.sum(time_per_chunk)
        print(f"===========\nDevice {device_id} processed {len(assigned_chunks)} chunks with average time per chunk: {avg_time:.4f} s\ntotal time: {total_time:.4f}")

        # return local_updates
        return local_out, local_wsum

    # run workers concurrently
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_gpus) as exc:
        for dev_id in range(n_gpus):
            assigned = per_device_chunks[dev_id]
            futures.append((dev_id, exc.submit(_worker, dev_id, assigned)))

        timetotal_start = time.time()
        pbar = tqdm(total=total_chunks, desc="hop-chunked 3D NUFFT reconstruction")
        # collect results as they finish and update progress
        for dev_id, fut in futures:
            assigned_count = len(per_device_chunks[dev_id])
            local_out, local_wsum = fut.result()
            out += local_out.get()
            wsum += local_wsum.get()
            # updates = fut.result()
            # # apply updates to host accumulators
            # for (tsl_out, hsl_out, wsl_out, weighted_chunk_np, weight_output_np) in updates:
            #     out[tsl_out, hsl_out, wsl_out] += weighted_chunk_np
            #     wsum[tsl_out, hsl_out, wsl_out] += weight_output_np
            pbar.update(assigned_count)
        timetotal_end = time.time()
        total_time = timetotal_end - timetotal_start
        print(f"Total reconstruction time across {n_gpus} devices: {total_time:.4f} s")
        pbar.close()

    flux = out / (wsum + 1e-12)
    return flux


def uwb1d_quanta(
    frames, alpha=None, T_exp=1.0, Nt=None, recon_slice: slice=None,
    cuda=False, load_frames_on_vram=False, load_output_on_vram=False
):
    """
    Implementation of per-pixel probing as desctibed in the original
    Passive Ultra-Wideband paper method. Optimized for quanta videos.
    """
    xp = cp if cuda else np
    if cuda and load_frames_on_vram:
        frames = cp.asarray(frames, dtype="uint8")
    else:
        frames = np.asarray(frames, dtype="uint8")
    T, H, W = frames.shape
    # leave on the cpu because this is huge
    if recon_slice is not None:
        out = np.zeros((len(range(*recon_slice.indices(Nt))), H, W), dtype="float32")
    else:
        out = np.zeros((Nt, H, W), dtype="float32")
    if cuda and load_output_on_vram:
        out = cp.asarray(out)
    ft = np.fft.rfftfreq(Nt, 1/Nt)
    if alpha is None:
        alpha = 1 / len(ft)
    # there's no way we can just do fftn on axis=0, memory will die
    pbar = tqdm(total=H * W, desc="1D UWB")
    for i in range(H):
        for j in range(W):
            timeslice = xp.asarray(frames[:, i, j])
            F = xp.fft.rfft(timeslice) / xp.sqrt(T_exp)
            bd = compute_amplitude_bound(xp.count_nonzero(timeslice), V=T_exp, alpha=alpha)
            F[xp.abs(F) < bd] = 0.0
            fluxtime = xp.fft.irfft(F, n=Nt, norm="forward").real / xp.sqrt(T_exp)
            if cuda and not load_output_on_vram:
                fluxtime = fluxtime.get()
            if recon_slice is not None:
                out[:, i, j] = fluxtime[recon_slice]
            else:
                out[:, i, j] = fluxtime
        pbar.update(W)
    pbar.close()
    if cuda and load_frames_on_vram:
        del frames  # free up VRAM just in case
    if cuda and load_output_on_vram:
        out = out.get()
    return np.clip(out, 0, None)


def uwb1d_cont(pixel_timeseries, tt, ft, T_exp, alpha=None, type3=True):
    """
    Implementation of per-pixel probing as desctibed in the original
    Passive Ultra-Wideband paper method. For continuous-time data.

    Returns:
        tuple:
        - rate_uwb: reconstructed rate, shape (len(tt), H, W)
        - F_all: Fourier coefficients, shape (H, W, len(ft))
        - bound_uwb: amplitude bounds, shape (H, W)
    """
    if type3:
        ft = np.asarray(ft, dtype=np.float64)
    else:
        raise NotImplementedError("type1 uwb1d is currently broken, use type3")
        # try:
        #     ft = int(ft)
        # except Exception:
        #     raise ValueError("ft must be the max frequency, not an array if type3=False")
    ntimes = len(tt)
    H = len(pixel_timeseries)
    W = len(pixel_timeseries[0])
    rate_uwb = np.zeros((ntimes, H, W))
    bound_uwb = np.zeros((H, W))
    if type3:
        F_all = np.zeros((H, W, len(ft)), dtype=np.complex128)
    else:
        F_all = np.zeros((H, W, ft + 1), dtype=np.complex128)
    # assume integer frequencies up to nyquist, to handle any floating point issues
    recon_nyquist = round(0.5 / np.min(np.diff(tt)))
    herm_dropped = np.all(ft >= 0)

    if alpha is None:
        if type3:
            n_freqs = len(ft)
        else:
            n_freqs = ft + 1
        alpha = 1 / n_freqs

    pbar = tqdm(total=H*W, desc="UWB per-pixel reconstructions")
    for i in range(H):
        for j in range(W):
            stamps = pixel_timeseries[i][j]
            npoints = len(stamps)
            if npoints == 0:
                pbar.update(1)
                continue
            if type3:
                F = ihpp_nufft3(stamps, ft=ft, V=T_exp)
            else:
                F = ihpp_nufft1(stamps, ft_max=ft, V=T_exp)
                F = F[ft:]
            F_all[i, j, :] = F
            if type3:
                n_freqs = len(ft)
            else:
                n_freqs = ft + 1
            bound = compute_amplitude_bound(npoints, T_exp, alpha=alpha)
            bound_uwb[i, j] = bound
            if type3:
                idx = np.where((np.abs(F) > bound) & (ft <= recon_nyquist))[0]
                if len(idx) == 0:
                    pbar.update(1)
                    continue
                ft_in = ft[idx]
                F_in = F[idx]
                if herm_dropped:
                    F_in = double_real_spectrum_3d(F_in, ft_in, recon_nyquist)
                rate_uwb[:, i, j] = ihpp_inufft3(ft_in, F_in, tt, V=T_exp).real
            else:
                idx = np.where((np.abs(F) < bound))[0]
                if len(idx) == 0:
                    pbar.update(1)
                    continue
                F_recon = F.copy()
                F_recon[idx] = 0.0
                F_recon[recon_nyquist:] = 0.0
                rate_uwb[:, i, j] = ihpp_inufft2(F_recon, tt, V=T_exp, F_full=False).real
            pbar.update(1)
    pbar.close()
    rate_uwb = np.clip(rate_uwb, 0, None)
    return rate_uwb, F_all, bound_uwb
