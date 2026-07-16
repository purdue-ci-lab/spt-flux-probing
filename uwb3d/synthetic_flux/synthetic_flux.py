import numpy as np

def flux_func_gaussian_ball(
    start_x: float,
    start_y: float,
    width: int,
    height: int,
    num_t: int,
    T_exp: float,
    sigma: float,
    max_amplitude: float,
    normalize: bool = True,
    vx: float = 0,
    vy: float = 0,
):
    """
    Gaussian blob moving with constant velocity.

    Motion model:
        x(t) = start_x + vx * frame_index(t)
        y(t) = start_y + vy * frame_index(t)

    where:
        frame_index(t) = t * (num_frames / T_exp)

    Units
    -----
    start_x, start_y : pixels
    vx, vy           : pixels / frame
    sigma            : pixels
    t                : seconds   (input to compute_flux)
    x, y             : normalized in [0,1] if normalize=True

    Parameters
    ----------
    start_x, start_y : float
        Initial blob center in pixel coordinates.
    width, height : int
        Spatial dimensions (pixels).
    T_exp : float
        Total exposure time in seconds.
    vx, vy : float
        Velocity components in pixels per frame.
    sigma : float
        Gaussian standard deviation in pixels.
    max_amplitude : float
        Peak photon flux.
    normalize : bool
        If True, input x,y,t are assumed normalized in [0,1].

    Returns
    -------
    compute_flux : callable
        Function f(x,y,t) returning photon flux.
    upper_bound : float
        Maximum possible flux value.
    """

    volume = width * height * T_exp
    scale = volume if normalize else 1.0

    def compute_flux(x, y, t):
        # work on copies
        x = np.asarray(x, dtype=float).copy()
        y = np.asarray(y, dtype=float).copy()
        t = np.asarray(t, dtype=float).copy()

        # convert normalized coordinates to physical units
        if normalize:
            x *= width           # → pixels
            y *= height          # → pixels
            t *= T_exp           # → seconds

        # continuous frame index
        frame_idx = t * num_t / T_exp

        x_t = start_x + vx * frame_idx
        y_t = start_y + vy * frame_idx

        return max_amplitude * scale * np.exp(
            -((x - x_t)**2 + (y - y_t)**2) / (2.0 * sigma**2)
        )

    return compute_flux, max_amplitude * scale

