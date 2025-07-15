import math
from typing import Sequence, Dict
import numpy as np
from scipy.interpolate import CubicSpline

# Fundamental constants
PI = math.pi
C = 3e8  # Speed of light (m/s)

# ----------------------------------------------------------------------------
# Raw constellation maps (Gray-coded, unnormalized)
# ----------------------------------------------------------------------------
# Define raw (unnormalized) constellation points for easy reuse
_RAW_QPSK = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex128)
_raw_level_map = {0: -3, 1: -1, 3: 1, 2: 3}
_RAW_QAM16 = np.array(
    [(_raw_level_map[(s>>2)&3] + 1j * _raw_level_map[s&3]) for s in range(16)],
    dtype=np.complex128
)

# Pilot symbol index (common for QPSK and QAM)
PILOT = 0

# Map modulation order to raw LUT
_CONSTELLATIONS: Dict[int, np.ndarray] = {
    4: _RAW_QPSK,
    16: _RAW_QAM16,
}




def bits_per_symbol(mod_complexity: int) -> int:
    """
    Compute number of bits per symbol for an M-ary modulation.
    Raises ValueError if M is not a power of two.
    """
    bps = int(math.log2(mod_complexity))
    if 2**bps != mod_complexity:
        raise ValueError(f"Unsupported modulation order: {mod_complexity}")
    return bps


def get_constellation(mod_complexity: int) -> np.ndarray:
    """
    Return the raw constellation points for given modulation order.
    """
    try:
        return _CONSTELLATIONS[mod_complexity]
    except KeyError:
        raise ValueError(f"Unsupported modulation order: {mod_complexity}")


def normalize_constellation(raw_const: np.ndarray) -> np.ndarray:
    """
    Normalize a raw constellation so its average symbol energy = 1.
    """
    Es = np.mean(np.abs(raw_const)**2)
    return raw_const / math.sqrt(Es)


def get_normalized_constellation(mod_complexity: int) -> np.ndarray:
    """
    Return a unit-energy normalized constellation for given modulation.
    """
    raw = get_constellation(mod_complexity)
    return normalize_constellation(raw)


def get_pilot_symbol(mod_complexity: int) -> complex:
    """
    Return the normalized pilot symbol for the given modulation.
    """
    const = get_normalized_constellation(mod_complexity)
    return const[PILOT]  # index defined above



def ebn0_to_snr_db(ebno_db: float, bps: int) -> float:
    """
    Convert Eb/N0 (dB) to Es/N0 (dB) given bits-per-symbol.
    Es/N0 (dB) = Eb/N0 (dB) + 10*log10(bps).
    """
    return ebno_db + 10 * math.log10(bps)


def snr_db_to_noise_sigma(snr_db: float) -> float:
    """
    Convert SNR in dB to per-quadrature Gaussian noise std dev,
    assuming unit symbol energy: N0 = 1 / (10^(SNR/10)), sigma = sqrt(N0/2).
    """
    n0 = 1.0 / (10 ** (snr_db / 10))
    return math.sqrt(n0 / 2)

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

def bit_error_rate(
    tx_bits: Sequence[int],
    rx_bits: Sequence[int]
) -> float:
    """
    Compute the Bit Error Rate (BER) between transmitted and received bit sequences.

    Parameters
    ----------
    tx_bits
        Original bit sequence (0/1).
    rx_bits
        Demodulated bit sequence (0/1), same length as tx_bits.

    Returns
    -------
    ber : float
        Ratio of bit errors to total bits.

    Raises
    ------
    ValueError
        If tx_bits and rx_bits lengths differ.
    """
    if len(tx_bits) != len(rx_bits):
        raise ValueError("tx_bits and rx_bits must have the same length")
    errors = np.count_nonzero(np.array(tx_bits) != np.array(rx_bits))
    return errors / len(tx_bits)