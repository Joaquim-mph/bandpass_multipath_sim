import numpy as np
import math
from typing import Tuple

from utils import QPSK, QAM16, PILOT


def generate_sequence_bins(
    mod_complexity: int,
    n_bits: int
) -> np.ndarray:
    """
    Generate a random symbol sequence from a bit stream.

    Parameters
    ----------
    mod_complexity
        Modulation order (e.g., 4 for QPSK, 16 for 16-QAM).
    n_bits
        Total number of bits to generate.

    Returns
    -------
    symbols : np.ndarray of int
        Array of symbol indices in [0, M-1]. Length = n_bits / log2(M).

    Raises
    ------
    ValueError
        If n_bits < log2(mod_complexity) or mod_complexity is not a power of two.
    """
    # bits per symbol
    bps = int(math.log2(mod_complexity))
    if 2**bps != mod_complexity:
        raise ValueError(f"Unsupported modulation order: {mod_complexity}")
    if n_bits < bps:
        raise ValueError(f"Need at least {bps} bits to form one symbol, got {n_bits}")

    max_sym = 2**bps - 1
    n_sym = n_bits // bps
    # generate uniformly random symbols
    symbols = np.random.randint(0, max_sym + 1, size=n_sym, dtype=int)
    return symbols


def modulate_sequence(
    sequence: np.ndarray,
    mod_complexity: int
) -> np.ndarray:
    """
    Map integer symbols to complex constellation points.

    Parameters
    ----------
    sequence
        Array of integer symbols [0..M-1].
    mod_complexity
        Modulation order (4 or 16 supported).

    Returns
    -------
    tx_symbols : np.ndarray of complex
        Complex baseband waveform with unit average symbol energy.
    """
    if mod_complexity == 4:
        lut, norm = QPSK, math.sqrt(2)
    elif mod_complexity == 16:
        lut, norm = QAM16, math.sqrt(10)
    else:
        raise ValueError(f"Modulation complexity {mod_complexity} not supported")

    # lookup + normalize
    return np.array([lut[s] / norm for s in sequence], dtype=complex)


def add_pilot_symbols(
    data: np.ndarray,
    mod_complexity: int,
    pilot_spacing: int
) -> np.ndarray:
    """
    Insert pilot symbols at regular intervals into a symbol stream.

    Parameters
    ----------
    data
        Transmit symbol array (complex). No pilots.
    mod_complexity
        Modulation order for pilot mapping.
    pilot_spacing
        Number of data symbols between consecutive pilots.

    Returns
    -------
    tx_with_pilots : np.ndarray of complex
        Symbol array including pilots every (pilot_spacing+1) positions.

    Raises
    ------
    ValueError
        If pilot_spacing < 1.
    """
    if pilot_spacing < 1:
        raise ValueError("pilot_spacing must be >= 1")

    # select pilot constellation point (normalized)
    if mod_complexity == 4:
        pilot = QPSK[PILOT] / math.sqrt(2)
    elif mod_complexity == 16:
        pilot = QAM16[PILOT] / math.sqrt(10)
    else:
        raise ValueError(f"Modulation complexity {mod_complexity} not supported")

    N = len(data)
    n_pilots = math.ceil(N / pilot_spacing) + 1
    out = []
    data_idx = 0

    for i in range(N + n_pilots):
        if i % (pilot_spacing + 1) == 0:
            out.append(pilot)
        else:
            out.append(data[data_idx])
            data_idx += 1

    return np.array(out, dtype=complex)
