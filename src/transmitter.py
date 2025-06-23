import numpy as np
import math
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


def modulate_sequence(sequence: np.ndarray, mod_complexity: int) -> np.ndarray:
    """
    Map integer symbols to complex constellation points via array indexing.
    Works with np.int64 or Python ints interchangeably.
    """
    if mod_complexity == 4:
        lut = QPSK
    elif mod_complexity == 16:
        lut = QAM16
    else:
        raise ValueError(f"Unsupported modulation order: {mod_complexity}")

    # direct NumPy indexing is safe under multiprocessing
    return lut[sequence]


def add_pilot_symbols(
    data: np.ndarray,
    mod_complexity: int,
    pilot_spacing: int
) -> np.ndarray:
    if pilot_spacing < 1:
        raise ValueError("pilot_spacing must be >= 1")

    # pick pilot point
    if mod_complexity == 4:
        pilot = QPSK[PILOT] / math.sqrt(2)
    elif mod_complexity == 16:
        pilot = QAM16[PILOT] / math.sqrt(10)
    else:
        raise ValueError(f"Unsupported modulation: {mod_complexity}")

    N = data.size
    step = pilot_spacing + 1
    # same formula for number of pilots
    n_pilots = math.ceil(N / pilot_spacing) + 1
    out_len = N + n_pilots

    # allocate output
    out = np.empty(out_len, dtype=complex)

    # indices where pilots go
    pilot_idx = np.arange(0, out_len, step)
    out[pilot_idx] = pilot

    # fill data into the other slots
    mask = np.ones(out_len, dtype=bool)
    mask[pilot_idx] = False
    out[mask] = data  

    return out

