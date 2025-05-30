import numpy as np
import math
from sequence_generator import (
    generate_sequence_bins,
    modulate_sequence,
    add_pilot_symbols,
    remove_pilot_symbols,
    separate_real_imaginary,
    add_noise,
    apply_channel,
    equalize_channel,
)

def main():
    rng = np.random.default_rng(seed=42)

    # 1) Generate bits → symbols
    bits = 1000
    seq = generate_sequence_bins(4, bits)
    assert seq.dtype == int
    assert seq.min() >= 0 and seq.max() < 4
    print("✔ sequence generation")

    # 2) Modulate → unit‐energy constellation
    mod = modulate_sequence(seq, 4)
    avg_energy = np.mean(np.abs(mod)**2)
    assert math.isclose(avg_energy, 1.0, rel_tol=1e-2)
    print(f"✔ modulation (avg energy = {avg_energy:.3f})")

    # 3) Insert & remove pilots
    pilots_in = add_pilot_symbols(mod, 4, pilot_spacing=10)
    payload, pilots_out = remove_pilot_symbols(pilots_in, pilot_spacing=10)
    # after remove, payload should equal original mod sequence
    assert payload.shape == mod.shape
    assert np.allclose(payload, mod)
    print("✔ pilot insert/remove")

    # 4) Split I/Q
    i, q = separate_real_imaginary(mod[:5])
    assert np.allclose(i + 1j*q, mod[:5])
    print("✔ I/Q separation")

    # 5) AWGN: measure post‐noise SNR
    snr_db = 10.0
    noisy = add_noise(mod, 4, snr_db, rng)
    # estimate SNR from sample variances
    signal_power = np.mean(np.abs(mod)**2)
    noise_power  = np.mean(np.abs(noisy-mod)**2)
    measured_snr = 10*np.log10(signal_power/noise_power)
    print(f"→ requested SNR={snr_db} dB, measured ≈{measured_snr:.1f} dB")
    assert abs(measured_snr - snr_db) < 2.0
    print("✔ AWGN")

    # 6) Simple flat‐fading channel + equalization
    H = (rng.normal(size=mod.shape) + 1j*rng.normal(size=mod.shape)) * math.sqrt(0.5)
    rx = apply_channel(mod, H)
    eq = equalize_channel(rx, H)
    assert np.allclose(eq, mod, atol=1e-6)
    print("✔ channel apply/equalize")

    print("\nAll tests passed!")

if __name__ == "__main__":
    main()
