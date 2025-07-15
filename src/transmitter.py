import numpy as np
from utils import bits_per_symbol, get_normalized_constellation, get_pilot_symbol
from typing import Sequence


def generate_sequence_bins(
    mod_complexity: int,
    n_bits: int
) -> np.ndarray:
    """
    Generate a random symbol sequence from a bit stream.
    """
    # bits per symbol
    bps = bits_per_symbol(mod_complexity)
    if n_bits < bps:
        raise ValueError(f"Need at least {bps} bits to form one symbol, got {n_bits}")

    max_sym = mod_complexity - 1
    n_sym = n_bits // bps
    return np.random.randint(0, max_sym + 1, size=n_sym, dtype=int)


def modulate_sequence(
    sequence: Sequence[int],
    mod_complexity: int
) -> np.ndarray:
    """
    Map integer symbols to unit-energy constellation points.
    """
    seq = np.asarray(sequence, dtype=int)
    if seq.min() < 0 or seq.max() >= mod_complexity:
        raise ValueError(f"Symbol index out of range for M={mod_complexity}")

    lut = get_normalized_constellation(mod_complexity)
    return lut[seq]



def add_pilot_symbols(
    data: np.ndarray,
    mod_complexity: int,
    pilot_spacing: int
) -> np.ndarray:
    """
    Insert pilot symbols regularly in the data stream.

    Pilots are placed before the first data symbol and after every
    `pilot_spacing` data symbols.
    """
    if pilot_spacing < 1:
        raise ValueError("pilot_spacing must be >= 1")

    pilot = get_pilot_symbol(mod_complexity)
    N = data.size
    n_pilots = (N + pilot_spacing - 1) // pilot_spacing + 1
    out_len = N + n_pilots

    out = np.empty(out_len, dtype=complex)
    step = pilot_spacing + 1
    pilot_idx = np.arange(0, out_len, step)
    out[pilot_idx] = pilot

    mask = np.ones(out_len, dtype=bool)
    mask[pilot_idx] = False
    out[mask] = data

    return out