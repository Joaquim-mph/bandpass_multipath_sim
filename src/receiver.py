import numpy as np
import math
import cmath
from typing import Sequence, List, Tuple
from numba import njit
from utils import bits_per_symbol, get_normalized_constellation

@njit
def equalize_channel(data: np.ndarray, H: np.ndarray) -> np.ndarray:
    if data.shape[0] != H.shape[0]:
        raise ValueError("data and H must have same shape for equalization")
    n = data.shape[0]
    out = np.empty(n, dtype=np.complex128)
    for i in range(n):
        out[i] = data[i] / H[i]
    return out

@njit
def separate_real_imaginary(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    Nearest‐neighbor demodulation using shared normalized LUT.
    """
    arr = np.asarray(data, dtype=np.complex128)
    lut = get_normalized_constellation(modulation_order)
    # distance matrix and argmin
    dists = np.abs(arr[:, None] - lut[None, :])
    idxs = dists.argmin(axis=1)
    return idxs.tolist()


@njit
def gray_to_binary(n: int) -> int:
    mask = n
    result = n
    while mask:
        mask >>= 1
        result ^= mask
    return result

@njit
def symbol_indices_to_bits(symbols: np.ndarray, bps: int) -> np.ndarray:
    n_sym = symbols.shape[0]
    bits  = np.empty(n_sym * bps, dtype=np.int64)
    for i in range(n_sym):
        # first convert Gray index to binary index
        g = symbols[i]
        bidx = gray_to_binary(g)
        # then unpack bits MSB→LSB
        for b in range(bps):
            bits[i*bps + b] = (bidx >> (bps - 1 - b)) & 1
    return bits

@njit
def bits_to_symbol_indices(bits: np.ndarray, bps: int) -> np.ndarray:
    n_bits = bits.shape[0]
    n_sym  = n_bits // bps
    syms   = np.empty(n_sym, np.int64)

    for i in range(n_sym):
        # pack
        val = 0
        for b in range(bps):
            val = (val << 1) | bits[i*bps + b]
        # binary → Gray
        syms[i] = val ^ (val >> 1)
    return syms

def remove_pilot_symbols(
    data: np.ndarray,
    pilot_spacing: int
) -> Tuple[np.ndarray, np.ndarray]:
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
