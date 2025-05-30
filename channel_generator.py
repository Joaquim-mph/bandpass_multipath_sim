import numpy as np
import math

PI = math.pi
C = 3e8  # speed of light (m/s)

def generate_doppler_mpth(
    size: int,
    paths: int,
    speed_kmh: float,
    carrier_freq: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate a time‐varying multipath fading channel using a sum‐of‐sinusoids (Jakes) model.
    
    Parameters
    ----------
    size
        Number of time samples.
    paths
        Number of reflected paths to sum.
    speed_kmh
        User speed in km/h.
    carrier_freq
        Carrier frequency in Hz.
    rng
        A numpy.random.Generator for reproducibility.
    
    Returns
    -------
    H : np.ndarray of shape (size,)
        Complex channel gains, normalized so E[|H|^2]=1.
    """
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



# if __name__ == "__main__":
#     rng = np.random.default_rng(123)
#     H_doppler  = generate_doppler_mpth(size=1024, paths=8,
#                                        speed_kmh=50,
#                                        carrier_freq=2.4e9,
#                                        rng=rng)
#     H_rayleigh = generate_rayleigh_mpth(size=1024, rng=rng)
#     # H_doppler and H_rayleigh are now numpy arrays of complex channel gains
