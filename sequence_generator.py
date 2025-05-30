import numpy as np
import math
import cmath
from typing import List, Sequence, Tuple
from scipy.interpolate import CubicSpline

PI = math.pi
C = 3e8  # speed of light (m/s)

# QPSK: 0→1+1j, 1→-1+1j, 2→-1-1j, 3→1-1j
QPSK = {
    0:  1+1j,
    1: -1+1j,
    2: -1-1j,
    3:  1-1j,
}

# 16-QAM Gray-coded 4×4 grid {–3,–1,+1,+3} in each axis
_level_map = {0: -3, 1: -1, 3: 1, 2: 3}
QAM16 = {
    s: (_level_map[(s >> 2) & 0x3] + 1j * _level_map[s & 0x3])
    for s in range(16)
}

PILOT = 0   # symbol index used for pilots



def generate_sequence_bins(mod_complexity: int, n_bits: int) -> np.ndarray:
    bps      = int(math.log2(mod_complexity))        # bits per symbol
    max_sym  = 2**bps - 1                             # largest symbol index
    n_sym    = n_bits // bps                          # how many symbols
    print(f"Sequence of size {n_sym} with max value {max_sym} "
          f"(bin {max_sym:0{bps}b})")
    # uniform integers in [0, max_sym]
    return np.random.randint(0, max_sym + 1, size=n_sym, dtype=int)


def modulate_sequence(
    sequence: np.ndarray,
    mod_complexity: int
) -> np.ndarray:
    if mod_complexity == 4:
        lut, norm = QPSK, math.sqrt(2)
    elif mod_complexity == 16:
        lut, norm = QAM16, math.sqrt(10)
    else:
        raise ValueError("Modulation complexity not supported")

    # lookup + per-symbol normalization
    return np.array([lut[s] / norm for s in sequence], dtype=complex)



def add_pilot_symbols(
    data: np.ndarray,
    mod_complexity: int,
    pilot_spacing: int,
) -> np.ndarray:
    # pick the pilot point
    if mod_complexity == 4:
        pilot = QPSK[PILOT] / math.sqrt(2)
    elif mod_complexity == 16:
        pilot = QAM16[PILOT] / math.sqrt(10)
    else:
        raise ValueError("Modulation complexity not supported")

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



def separate_real_imaginary(
    data: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return data.real.copy(), data.imag.copy()



def add_noise(
    data: np.ndarray,
    snr_db: float,
    rng: np.random.Generator
) -> np.ndarray:
    # N0 = 1 / (10^(SNR/10)), each quadrature has var = N0/2
    N0    = 1.0 / (10 ** (snr_db / 10))
    sigma = math.sqrt(N0 / 2)
    # generate complex Gaussian noise
    noise = (rng.normal(0, sigma, size=data.shape) +
             1j * rng.normal(0, sigma, size=data.shape))
    # optional debug print of last noise sample
    print(f"SNR={snr_db}dB, last noise sample = {noise[-1].real:.3e}+{noise[-1].imag:.3e}j")
    return data + noise



def apply_channel(
    data: np.ndarray,
    H: np.ndarray
) -> np.ndarray:
    return data * H



def equalize_channel(
    data: np.ndarray,
    H: np.ndarray
) -> np.ndarray:
    if data.shape != H.shape:
        raise ValueError("data and H must have same shape")
    return data / H



def remove_pilot_symbols(
    data: np.ndarray,
    pilot_spacing: int
) -> tuple[np.ndarray, np.ndarray]:
    payload = []
    pilots  = []
    for i, x in enumerate(data):
        if i % (pilot_spacing + 1) == 0:
            pilots.append(x)
        else:
            payload.append(x)
    return np.array(payload, dtype=complex), np.array(pilots, dtype=complex)



def generate_doppler_mpth(
    size: int,
    paths: int,
    speed_kmh: float,
    carrier_freq: float,
    rng: np.random.Generator
) -> np.ndarray:
    # wavelength and max Doppler shift
    wavelength   = C / carrier_freq
    max_doppler  = (speed_kmh / 3.6) / wavelength

    # normalized time axis t ∈ [0,1)
    t = np.arange(size) / size

    # equal‐power per path
    an = 1.0 / math.sqrt(paths)

    # random initial phases θₙ ∈ [0,2π)
    thetan = rng.uniform(0, 2*PI, size=paths)

    # random Doppler offsets fₙ = f_max * cos(2π·φₙ)
    phi    = rng.uniform(0, 2*PI, size=paths)
    fDn    = max_doppler * np.cos(2 * PI * phi)

    # build the channel H(t) = Σₙ an·e^{j(θₙ − 2π fₙ t)}
    H = np.zeros(size, dtype=complex)
    for n in range(paths):
        phase = thetan[n] - 2 * PI * fDn[n] * t
        H   += an * np.exp(1j * phase)

    return H


def generate_rayleigh_mpth(
    size: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate a memoryless Rayleigh fading channel (i.i.d. per‐sample).
    
    Parameters
    ----------
    size
        Number of samples.
    rng
        A numpy.random.Generator for reproducibility.
    
    Returns
    -------
    H : np.ndarray of shape (size,)
        Complex Gaussian samples ~ CN(0,1).
    """
    # each quadrature ~ N(0,½) => amplitude Rayleigh, power=1
    real = rng.normal(0.0, 1.0, size=size)
    imag = rng.normal(0.0, 1.0, size=size)
    return (real + 1j*imag) * math.sqrt(0.5)




def QAMdemod(
    I_matrix: List[List[float]],
    Q_matrix: List[List[float]],
    modulation_order: int
) -> List[List[int]]:

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
    mtype = modulation_type.upper()
    if mtype == "QAM":
        I, Q = separate_real_imaginary(data)
        return qamdemod_interface(I, Q, modulation_order)
    elif mtype == "PSK":
        return pskdemod(data, modulation_order)
    else:
        print("Unrecognized Modulation Type")
        return [-1]



def fft_interpolate_complex(original: np.ndarray, n_interp: int) -> np.ndarray:
    n = original.size
    if n <= 0:
        raise ValueError("Input vector size must be greater than zero.")
    # New FFT length
    intersize = n + n * n_interp

    # FFT of original
    X = np.fft.fft(original)

    # Zero-pad in frequency domain
    Y = np.zeros(intersize, dtype=complex)
    half_n = n // 2

    # Copy low-frequency (including DC and Nyquist)
    Y[: half_n + 1] = X[: half_n + 1]
    # Copy high-frequency components to the end to maintain hermitian symmetry
    for i in range(1, half_n):
        Y[-i] = X[-i]

    # Inverse FFT and renormalize amplitude
    y = np.fft.ifft(Y) * (intersize / n)

    # Extract only the newly interpolated samples:
    # we skip each original sample (every n_interp-th point in the expanded array)
    output_len = intersize - n_interp - n
    out = np.empty(output_len, dtype=complex)
    idx_shift = 0
    for i in range(output_len):
        if i % n_interp == 0:
            idx_shift += 1
        out[i] = y[i + idx_shift]

    return out


def interpolate_complex_points(point1: complex, point2: complex, n_interp: int) -> np.ndarray:
    # Parameter axis
    x = np.array([0.0, 1.0])
    # Fit two separate 1D cubics on the real and imag parts
    cs_real = CubicSpline(x, np.array([point1.real, point2.real]))
    cs_imag = CubicSpline(x, np.array([point1.imag, point2.imag]))

    xi = np.linspace(0, 1, n_interp)
    return cs_real(xi) + 1j * cs_imag(xi)

