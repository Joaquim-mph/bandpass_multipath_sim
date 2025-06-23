import numpy as np
import math
import cmath
from typing import Sequence, List, Tuple
from numba import njit
from utils import QPSK, QAM16, bits_per_symbol


@njit
def equalize_channel(data: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Zero-forcing equalizer JIT-compiled with Numba.
    Raises ValueError if shapes mismatch.
    """
    # shape check
    if data.shape[0] != H.shape[0]:
        raise ValueError("data and H must have same shape for equalization")

    n = data.shape[0]
    eq = np.empty(n, dtype=np.complex128)
    for i in range(n):
        eq[i] = data[i] / H[i]
    return eq


@njit
def separate_real_imaginary(data: np.ndarray) -> tuple:
    """
    Split complex array into real and imaginary float64 arrays.
    """
    n = data.shape[0]
    I = np.empty(n, dtype=np.float64)
    Q = np.empty(n, dtype=np.float64)
    for i in range(n):
        I[i] = data[i].real
        Q[i] = data[i].imag
    return I, Q



def demod(
    data: Sequence[complex],
    modulation_order: int,
    modulation_type: str
) -> List[int]:
    """
    Dispatch‐style demod using fast NumPy nearest‐neighbor.
    Pass modulation_type='PSK' or 'QAM'.
    """
    arr = np.asarray(data, dtype=np.complex128)
    if modulation_type.upper() == 'PSK':
        lut = QPSK
    elif modulation_type.upper() == 'QAM':
        if modulation_order != len(QAM16):
            raise ValueError("Unsupported QAM order")
        lut = QAM16
    else:
        raise ValueError(f"Unknown modulation type '{modulation_type}'")

    # NxM distance matrix, then argmin over axis=1
    # (this is pure NumPy, super‐fast, and handles any integer dtype)
    dists = np.abs(arr[:, None] - lut[None, :])
    idxs  = dists.argmin(axis=1)
    return idxs.tolist()


@njit
def symbol_indices_to_bits(symbols: np.ndarray, bps: int) -> np.ndarray:
    """
    Convert symbol indices to bits (MSB-first) in JIT.
    """
    n_sym = symbols.shape[0]
    bits = np.empty(n_sym * bps, dtype=np.int64)
    for i in range(n_sym):
        sym = symbols[i]
        for b in range(bps):
            bits[i*bps + b] = (sym >> (bps - 1 - b)) & 1
    return bits


@njit
def bits_to_symbol_indices(bits: np.ndarray, bps: int) -> np.ndarray:
    """
    Convert bits back to symbol indices in JIT.
    """
    n_bits = bits.shape[0]
    n_sym = n_bits // bps
    syms = np.empty(n_sym, dtype=np.int64)
    for i in range(n_sym):
        val = 0
        for b in range(bps):
            val = (val << 1) | bits[i*bps + b]
        syms[i] = val
    return syms


def remove_pilot_symbols(
    data: np.ndarray,
    pilot_spacing: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract payload and pilot symbols from a stream with regularly inserted pilots.

    Pilots are assumed inserted every (pilot_spacing+1)-th sample, starting at index 0.

    Parameters
    ----------
    data
        Received symbol stream including pilots.
    pilot_spacing
        Number of data symbols between consecutive pilots.

    Returns
    -------
    payload : np.ndarray of complex
        Data symbols with pilots removed.
    pilots : np.ndarray of complex
        Pilot symbols in the order they appeared.

    Raises
    ------
    ValueError
        If pilot_spacing < 1.
    """
    if pilot_spacing < 1:
        raise ValueError("pilot_spacing must be >= 1")

    payload = []
    pilots = []
    for idx, sample in enumerate(data):
        if idx % (pilot_spacing + 1) == 0:
            pilots.append(sample)
        else:
            payload.append(sample)

    return np.array(payload, dtype=complex), np.array(pilots, dtype=complex)

