import numpy as np
import math
from typing import Tuple, Optional

from utils import PI, C, snr_db_to_noise_sigma


def apply_awgn(
    data: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    verbose: bool = False
) -> np.ndarray:
    """
    Add complex AWGN to a signal to achieve the specified SNR (in dB).

    Parameters
    ----------
    data
        Input complex baseband signal.
    snr_db
        Desired SNR in dB (Es/N0, assuming unit symbol energy).
    rng
        Numpy random Generator for reproducibility.
    verbose
        If True, prints the last noise sample for debugging.

    Returns
    -------
    noisy_signal : np.ndarray
        data + complex Gaussian noise with per-quadrature sigma.
    """
    sigma = snr_db_to_noise_sigma(snr_db)
    noise = rng.normal(0.0, sigma, size=data.shape) + 1j * rng.normal(0.0, sigma, size=data.shape)
    if verbose:
        nr, ni = noise[-1].real, noise[-1].imag
        print(f"AWGN: SNR={snr_db} dB, last noise sample = {nr:.3e}+{ni:.3e}j")
    return data + noise


def apply_channel(
    data: np.ndarray,
    H: np.ndarray
) -> np.ndarray:
    """
    Apply a per-symbol complex channel gain H to a signal.

    Parameters
    ----------
    data
        Transmit symbol array (complex).
    H
        Channel gain array of same shape as data.

    Returns
    -------
    received : np.ndarray
        data * H

    Raises
    ------
    ValueError
        If data and H shapes mismatch.
    """
    if data.shape != H.shape:
        raise ValueError("data and H must have same shape for channel application")
    return data * H


def generate_rayleigh_mpth(
    size: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate memoryless Rayleigh fading channel coefficients.

    Parameters
    ----------
    size
        Number of channel samples to generate.
    rng
        Numpy random Generator for reproducibility.

    Returns
    -------
    H : np.ndarray
        Complex Gaussian samples ~ CN(0,1) (unit average power).
    """
    sigma = math.sqrt(0.5)
    real = rng.normal(0.0, sigma, size=size)
    imag = rng.normal(0.0, sigma, size=size)
    return real + 1j*imag


def generate_doppler_mpth(
    size: int,
    paths: int,
    speed_kmh: float,
    carrier_freq: float,
    rng: np.random.Generator
) -> np.ndarray:
    if size <= 0 or paths <= 0 or carrier_freq <= 0:
        raise ValueError("size, paths, and carrier_freq must be > 0")

    # Precompute constants
    wavelength   = C / carrier_freq
    max_doppler  = (speed_kmh / 3.6) / wavelength
    t            = np.arange(size) / size          
    an           = 1.0 / math.sqrt(paths)

    # Draw all ray phases & Doppler shifts at once
    thetan = rng.uniform(0.0, 2*PI, size=paths)
    phi    = rng.uniform(0.0, 2*PI, size=paths)
    fDn    = max_doppler * np.cos(2 * PI * phi)

    # Build a (paths × size) array of phases:
    phases = thetan[:,None] - 2 * PI * fDn[:,None] * t[None,:] 

    # Sum across rays and apply amplitude
    H = an * np.exp(1j * phases).sum(axis=0)  

    return H


def transmit_through_channel(
    data: np.ndarray,
    model: str,
    snr_db: float,
    rng: np.random.Generator,
    paths: Optional[int] = None,
    speed_kmh: Optional[float] = None,
    carrier_freq: Optional[float] = None,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Send data through a specified channel model + AWGN, returning received signal and true H.

    Parameters
    ----------
    data
        Transmit symbol array.
    model
        Channel model: 'awgn', 'rayleigh', or 'doppler'.
    snr_db
        AWGN SNR in dB.
    rng
        Random generator.
    paths, speed_kmh, carrier_freq
        Required for 'doppler' model.
    verbose
        If True, print debug info.

    Returns
    -------
    rx : np.ndarray
        Received symbols after channel + noise.
    H  : np.ndarray
        True channel gains applied.
    """
    model = model.lower()
    
    if model == 'awgn':
        H = np.ones_like(data)
        rx = apply_awgn(data, snr_db, rng, verbose)
    else:
        if model == 'rayleigh':
            H = generate_rayleigh_mpth(len(data), rng)
        elif model == 'doppler':
            if None in (paths, speed_kmh, carrier_freq):
                raise ValueError("paths, speed_kmh, and carrier_freq must be provided for doppler model")
            H = generate_doppler_mpth(len(data), paths, speed_kmh, carrier_freq, rng)
        else:
            raise ValueError(f"Unknown channel model '{model}'")
    rx = apply_channel(data, H)
    rx = apply_awgn(rx, snr_db, rng, verbose)
    return rx, H
