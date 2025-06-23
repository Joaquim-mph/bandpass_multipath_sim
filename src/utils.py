import math
from typing import Dict
import numpy as np

# Fundamental constants
PI = math.pi
C = 3e8  # Speed of light (m/s)

# ----------------------------------------------------------------------------
# Constellation maps (Gray-coded)
# ----------------------------------------------------------------------------
# QPSK: 0→1+1j, 1→-1+1j, 2→-1-1j, 3→1-1j
# 4‐PSK (QPSK) Gray‐map: 0→1+1j, 1→-1+1j, 2→-1-1j, 3→1-1j
QPSK = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex128) / math.sqrt(2)

# 16-QAM Gray‐map (4×4): levels {–3, –1, +1, +3}
_level_map = {0: -3, 1: -1, 3: 1, 2: 3}
QAM16 = np.array(
    [(_level_map[(s>>2)&3] + 1j*_level_map[s&3]) for s in range(16)],
    dtype=np.complex128
) / math.sqrt(10)


# Pilot symbol index (common for QPSK and QAM)
PILOT = 0


def bits_per_symbol(mod_complexity: int) -> int:
    """
    Compute number of bits per symbol for an M-ary modulation.
    Raises ValueError if M is not a power of two.
    """
    bps = int(math.log2(mod_complexity))
    if 2**bps != mod_complexity:
        raise ValueError(f"Unsupported modulation order: {mod_complexity}")
    return bps


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
