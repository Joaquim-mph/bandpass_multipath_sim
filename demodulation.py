import math
import cmath
from typing import List, Sequence, Tuple

# ----------------------------------------------------------------------
# You’ll need these to exist in your Python module:
#   • QAMdemod(I_matrix, Q_matrix, modulation_order) → List[List[int]]
#   • QPSK: a mapping from symbol index → complex point, e.g.
#         QPSK = {0: 1+1j, 1: -1+1j, 2: -1-1j, 3: 1-1j}
#   • separate_real_imaginary(data) → (I_list, Q_list)
# ----------------------------------------------------------------------



def QAMdemod(
    I_matrix: List[List[float]],
    Q_matrix: List[List[float]],
    modulation_order: int
) -> List[List[int]]:
    """
    Demodulate M‑QAM by mapping each (I,Q) pair to the nearest
    constellation point in a normalized square QAM grid.
    
    Parameters
    ----------
    I_matrix, Q_matrix : List of rows of I/Q samples
    modulation_order   : M (must be a perfect square, e.g. 16, 64)
    
    Returns
    -------
    List of rows of integer symbol indices [0 … M-1]
    """
    # Ensure M is a square number
    M = modulation_order
    m = int(math.sqrt(M))
    if m*m != M:
        raise ValueError("modulation_order must be a perfect square")
    
    # Generate the unnormalized constellation grid levels
    levels = list(range(-m+1, m, 2))  # e.g. for m=4 → [-3, -1, +1, +3]
    
    # Build and normalize the constellation
    # Average symbol energy of square QAM is 2*(M-1)/3
    energy = 2*(M-1)/3
    norm_factor = math.sqrt(energy)
    
    constellation = [
        complex(x, y) / norm_factor
        for y in reversed(levels)
        for x in levels
    ]
    
    # Demodulate each row
    demodulated_rows: List[List[int]] = []
    for I_row, Q_row in zip(I_matrix, Q_matrix):
        row_indices: List[int] = []
        for i_val, q_val in zip(I_row, Q_row):
            sample = complex(i_val, q_val)
            # find the constellation point with minimal Euclidean distance
            closest_idx = min(
                range(M),
                key=lambda k: abs(sample - constellation[k])
            )
            row_indices.append(closest_idx)
        demodulated_rows.append(row_indices)
    
    return demodulated_rows


def qamdemod_interface(
    I: Sequence[float],
    Q: Sequence[float],
    modulation_order: int
) -> List[int]:
    """
    Wraps your QAMdemod routine—which expects 2D I/Q arrays—and
    corrects a sign flip on Q before demodulating.
    """
    # pack into single-row “matrices”
    I_matrix = [list(I)]
    Q_matrix = [[-q for q in Q]]   # mirror Q to fix Gray‐code orientation

    # call the existing QAM demodulator
    received_seq: List[List[int]] = QAMdemod(I_matrix, Q_matrix, modulation_order)

    # flatten the first (and only) row
    return list(received_seq[0])


def pskdemod(
    data: Sequence[complex],
    modulation_order: int
) -> List[int]:
    """
    Demodulate an M‐PSK stream by slicing the circle into 2π/M sectors
    around each ideal PSK constellation point.
    """
    out: List[int] = []
    step = math.pi / modulation_order

    for value in data:
        angle = cmath.phase(value)
        # find which PSK symbol’s phase sector this sample falls into
        for idx, symbol in QPSK.items():
            sym_angle = cmath.phase(symbol)
            if sym_angle - step < angle <= sym_angle + step:
                out.append(idx)
                break

    return out


def demod(
    data: Sequence[complex],
    modulation_order: int,
    modulation_type: str
) -> List[int]:
    """
    Unified entry: dispatches to QAM or PSK demodulator based on the type string.
    """
    mtype = modulation_type.upper()
    if mtype == "QAM":
        I, Q = separate_real_imaginary(data)
        return qamdemod_interface(I, Q, modulation_order)
    elif mtype == "PSK":
        return pskdemod(data, modulation_order)
    else:
        print("Unrecognized Modulation Type")
        return [-1]
