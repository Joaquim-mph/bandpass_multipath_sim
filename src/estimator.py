import numpy as np
from scipy.interpolate import CubicSpline



def fft_interpolate_complex(
    pilot_samples: np.ndarray,
    pilot_spacing: int
) -> np.ndarray:
    """
    Interpolate a pilot-only channel estimate via FFT zero-padding.

    Given complex pilot_samples (H at pilot positions) and the spacing P,
    this function zero-pads its spectrum by factor (P+1) and returns
    all interpolated H values between pilots (excluding pilot slots).

    Parameters
    ----------
    pilot_samples
        Array of complex channel gains at pilot locations.
    pilot_spacing
        Number of data symbols between pilots.

    Returns
    -------
    h_interp : np.ndarray of complex
        Interpolated channel estimate for all non-pilot positions,
        length = len(pilot_samples) * pilot_spacing.

    Raises
    ------
    ValueError
        If pilot_samples is empty or pilot_spacing < 1.
    """
    n = pilot_samples.size
    if n == 0:
        raise ValueError("pilot_samples must be non-empty")
    if pilot_spacing < 1:
        raise ValueError("pilot_spacing must be >= 1")

    # FFT of pilot-only sequence
    X = np.fft.fft(pilot_samples)
    # Zero-pad in frequency domain
    intersize = n * (pilot_spacing + 1)
    Y = np.zeros(intersize, dtype=complex)
    half = n // 2
    # copy DC to Nyquist
    Y[: half+1] = X[: half+1]
    # copy upper frequencies to maintain Hermitian symmetry
    Y[-half+1:] = X[-half+1:]
    # inverse FFT and normalize
    y = np.fft.ifft(Y) * (intersize / n)
    # remove pilot positions, keep only interpolated samples
    return np.delete(y, np.arange(0, intersize, pilot_spacing+1))


def interpolate_complex_points(
    point1: complex,
    point2: complex,
    n_interp: int
) -> np.ndarray:
    """
    Generate a smooth cubic-spline interpolation between two complex points.

    Parameters
    ----------
    point1, point2
        Complex endpoints at x=0 and x=1.
    n_interp
        Total number of output samples, including endpoints.

    Returns
    -------
    np.ndarray
        Interpolated complex values of length n_interp.

    Raises
    ------
    ValueError
        If n_interp < 2.
    """
    if n_interp < 2:
        raise ValueError("n_interp must be >= 2 to include both endpoints")

    # real and imaginary cubic splines
    x = np.array([0.0, 1.0])
    cs_r = CubicSpline(x, np.array([point1.real, point2.real]))
    cs_i = CubicSpline(x, np.array([point1.imag, point2.imag]))

    xi = np.linspace(0.0, 1.0, n_interp)
    return cs_r(xi) + 1j * cs_i(xi)
