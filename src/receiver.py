import numpy as np
import math
import cmath
from typing import Sequence, List, Tuple

from utils import QPSK, bits_per_symbol


def equalize_channel(
    data: np.ndarray,
    H: np.ndarray
) -> np.ndarray:
    """
    Zero-forcing equalizer: divide received symbols by channel gains.

    Parameters
    ----------
    data
        Received symbol array (complex).
    H
        Channel gain array, same shape as data.

    Returns
    -------
    eq_data : np.ndarray
        Equalized symbol array.

    Raises
    ------
    ValueError
        If data and H shapes mismatch.
    """
    if data.shape != H.shape:
        raise ValueError("data and H must have same shape for equalization")
    return data / H


def separate_real_imaginary(
    data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split complex symbols into I and Q components.

    Parameters
    ----------
    data
        Complex symbol array.

    Returns
    -------
    I, Q : Tuple of np.ndarray
        Real and imaginary parts of input.
    """
    return data.real.copy(), data.imag.copy()


def QAMdemod(
    I_matrix: List[List[float]],
    Q_matrix: List[List[float]],
    modulation_order: int
) -> List[List[int]]:
    """
    Nearest-neighbor demodulation for square M-QAM.

    Parameters
    ----------
    I_matrix, Q_matrix
        2D lists of I and Q samples (rows of vectors).
    modulation_order
        M (perfect square, e.g. 4, 16, 64).

    Returns
    -------
    demodulated_rows : List of lists of ints
        Symbol indices [0..M-1] for each row.

    Raises
    ------
    ValueError
        If modulation_order is not a perfect square.
    """
    M = modulation_order
    m = int(math.sqrt(M))
    if m*m != M:
        raise ValueError("modulation_order must be a perfect square")

    # Constellation grid levels
    levels = list(range(-m+1, m, 2))
    # Average QAM energy = 2*(M-1)/3
    norm = math.sqrt(2*(M-1)/3)
    constellation = [complex(x, y)/norm for y in reversed(levels) for x in levels]

    demod_rows: List[List[int]] = []
    for I_row, Q_row in zip(I_matrix, Q_matrix):
        idx_row: List[int] = []
        for i_val, q_val in zip(I_row, Q_row):
            sample = complex(i_val, q_val)
            # find closest constellation point
            idx = min(range(M), key=lambda k: abs(sample - constellation[k]))
            idx_row.append(idx)
        demod_rows.append(idx_row)
    return demod_rows


def qamdemod_interface(
    I: Sequence[float],
    Q: Sequence[float],
    modulation_order: int
) -> List[int]:
    """
    Wrapper for QAMdemod that takes 1D I/Q vectors and flips Q-axis.
    """
    I_mat = [list(I)]
    Q_mat = [[-q for q in Q]]  # correct for mirrored Gray code
    rows = QAMdemod(I_mat, Q_mat, modulation_order)
    return rows[0]


def pskdemod(
    data: Sequence[complex],
    modulation_order: int
) -> List[int]:
    """
    Sector-based demodulation for M-PSK.
    """
    bps = bits_per_symbol(modulation_order)
    step = math.pi / modulation_order
    idxs: List[int] = []
    for sample in data:
        angle = cmath.phase(sample)
        for idx, sym in QPSK.items():
            sym_ang = cmath.phase(sym)
            if sym_ang - step < angle <= sym_ang + step:
                idxs.append(idx)
                break
    return idxs


def demod(
    data: Sequence[complex],
    modulation_order: int,
    modulation_type: str
) -> List[int]:
    """
    Dispatch to QAM or PSK demod based on modulation_type.
    """
    mtype = modulation_type.upper()
    if mtype == 'QAM':
        I, Q = separate_real_imaginary(np.array(data, dtype=complex))
        return qamdemod_interface(I, Q, modulation_order)
    elif mtype == 'PSK':
        return pskdemod(data, modulation_order)
    else:
        raise ValueError(f"Unknown modulation type '{modulation_type}'")


def symbol_indices_to_bits(
    symbols: Sequence[int],
    modulation_order: int
) -> List[int]:
    """
    Convert symbol indices to bit array (MSB-first) using Gray mapping.

    Parameters
    ----------
    symbols
        Sequence of symbol indices [0..M-1].
    modulation_order
        M (power of two).

    Returns
    -------
    bits : List[int]
        Flattened bit list.
    """
    bps = bits_per_symbol(modulation_order)
    bits: List[int] = []
    for sym in symbols:
        for b in range(bps-1, -1, -1):
            bits.append((sym >> b) & 1)
    return bits
