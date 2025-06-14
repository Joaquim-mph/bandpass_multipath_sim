import numpy as np
import math
from typing import Sequence, List, Tuple
from scipy.special import erfc
from utils import bits_per_symbol, ebn0_to_snr_db


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


def theoretical_ber_awgn(
    modulation_order: int,
    ebno_db: float
) -> float:
    """
    Compute theoretical BER over AWGN for square M-QAM or M-PSK.

    For QPSK, uses Q-function: BER = 0.5*erfc(sqrt(Eb/N0)).
    For M-QAM, approximates BER with union bound:
      BER ≈ (4*(1 - 1/√M)/log2(M)) * Q(√(3*log2(M)/(M-1) * Eb/N0)).

    Parameters
    ----------
    modulation_order
        M (4 for QPSK, 16, 64, ... for QAM) or M-PSK.
    ebno_db
        Eb/N0 in dB.

    Returns
    -------
    ber : float
        Theoretical bit-error rate.
    """
    bps = bits_per_symbol(modulation_order)
    ebno = 10**(ebno_db/10)
    # QPSK special case (same as BPSK)
    if modulation_order == 4:
        return 0.5 * erfc(math.sqrt(ebno))
    # square QAM
    m = int(math.sqrt(modulation_order))
    if m*m == modulation_order:
        # union bound approximation
        alpha = 4 * (1 - 1/m) / bps
        beta = 3 * bps / (modulation_order - 1)
        return alpha * 0.5 * erfc(math.sqrt(beta * ebno))
    # fallback PSK
    return 0.5 * erfc(math.sqrt(2 * ebno) * math.sin(math.pi / modulation_order))


def theoretical_ber_rayleigh(
    modulation_order: int,
    ebno_db: float
) -> float:
    """
    Theoretical BER over flat Rayleigh fading channel (coherent detection).

    Uses known closed-form for BPSK/QPSK:
      BER = 0.5 * (1 - sqrt(ebno/(1+ebno))).

    For higher-order QAM/PSK, this is approximate via averaging AWGN BER over exponential distribution.

    Parameters
    ----------
    modulation_order
        M (typically 4 for QPSK/BPSK).
    ebno_db
        Eb/N0 in dB.

    Returns
    -------
    ber : float
        Theoretical bit-error rate.
    """
    ebno = 10**(ebno_db/10)
    if modulation_order == 4:
        # QPSK ~ BPSK
        return 0.5 * (1 - math.sqrt(ebno / (1 + ebno)))
    # approximate for square QAM (m>2)
    # use numeric integration or union bound – here use AWGN formula as rough estimate
    return theoretical_ber_awgn(modulation_order, ebno_db)
