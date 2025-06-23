from transmitter import *
from receiver import *
from channel import *
from estimator import *
from utils import QPSK,QAM16, PILOT
import math
import numpy as np


# Test routines transmitter.py
def test_generate_sequence_bins():
    print("Testing generate_sequence_bins...")
    for mod in [4, 16]:
        for n_bits in [16, 32]:
            seq = generate_sequence_bins(mod, n_bits)
            expected_len = n_bits // int(math.log2(mod))
            assert len(seq) == expected_len, "Length mismatch"
            assert seq.min() >= 0 and seq.max() < mod, "Symbol out of range"
    print("  ✓ generate_sequence_bins passed")


def test_modulate_sequence():
    print("Testing modulate_sequence...")
    seq = np.array([0, 1, 2, 3])
    tx = modulate_sequence(seq, 4)
    avg_energy = np.mean(np.abs(tx)**2)
    assert np.isclose(avg_energy, 1.0, atol=1e-6), f"Avg energy not 1, got {avg_energy}"
    print(f"  ✓ modulate_sequence passed (avg energy = {avg_energy:.3f})")


def test_add_pilot_symbols():
    print("Testing add_pilot_symbols...")
    data = np.array([1+0j, 2+0j, 3+0j, 4+0j])
    out = add_pilot_symbols(data, 4, 2)
    pilot = QPSK[PILOT] / math.sqrt(2)
    expected = [pilot, data[0], data[1], pilot, data[2], data[3], pilot]
    assert np.allclose(out, expected), f"Pilot insertion mismatch: {out} vs {expected}"
    print("  ✓ add_pilot_symbols passed")
    

# Test receiver
def test_equalize_channel():
    print("Testing equalize_channel...")
    data = np.array([1+2j, 2+4j, 3+6j])
    H = np.array([1+1j, 2+0j, 3-3j])
    eq = equalize_channel(data, H)
    assert np.allclose(eq, data/H), "Equalization incorrect"
    try:
        equalize_channel(data, H[:-1])
    except ValueError:
        print("  ✓ Shape mismatch error raised")
    else:
        raise AssertionError("Shape mismatch error not raised")
    print("  ✓ equalize_channel passed")


def test_separate_real_imaginary():
    print("Testing separate_real_imaginary...")
    data = np.array([1+2j, -1-0.5j, 0+1j])
    I, Q = separate_real_imaginary(data)
    assert np.array_equal(I, np.array([1, -1, 0])), "I component mismatch"
    assert np.array_equal(Q, np.array([2, -0.5, 1])), "Q component mismatch"
    print("  ✓ separate_real_imaginary passed")


def test_qam_demod_roundtrip():
    print("Testing QAM demodulation round-trip…")
    # pick 10 random 16‑QAM symbol indices
    sym = np.random.randint(0, 16, size=10)
    # modulate via the official modulator
    tx = modulate_sequence(sym, 16)
    # demodulate via the generic demod API
    recovered = demod(tx, 16, 'QAM')
    assert recovered == sym.tolist(), f"QAM demod failed: {recovered} != {sym.tolist()}"
    print("  ✓ QAM round-trip demod passed")

def test_psk_demod_roundtrip():
    print("Testing PSK demodulation round-trip…")
    sym = np.random.randint(0, 4, size=10)
    tx = modulate_sequence(sym, 4)
    recovered = demod(tx, 4, 'PSK')
    assert recovered == sym.tolist(), f"PSK demod failed: {recovered} != {sym.tolist()}"
    print("  ✓ PSK round-trip demod passed")



def test_bit_symbol_conversion():
    print("Testing bit-symbol conversions...")
    bits = np.random.randint(0, 2, size=32)
    mod  = 16

    # round‐trip bits → symbols → bits
    syms    = bits_to_symbol_indices(bits, mod)
    bits_rt = symbol_indices_to_bits(syms, mod)

    # if symbol_indices_to_bits returns an ndarray, turn it into a list
    if isinstance(bits_rt, np.ndarray):
        bits_rt = bits_rt.tolist()

    assert bits_rt == bits.tolist(), "Bit-symbol round-trip failed"
    print("  ✓ bit-symbol conversion passed")


# test channel
def test_snr_db_to_noise_sigma():
    print("Testing snr_db_to_noise_sigma...")
    assert math.isclose(snr_db_to_noise_sigma(0), math.sqrt(1/2), rel_tol=1e-6)
    assert math.isclose(snr_db_to_noise_sigma(3), math.sqrt(0.5 / 10**(3/10)), rel_tol=1e-6)
    print("  ✓ snr_db_to_noise_sigma passed")

def test_apply_awgn():
    print("Testing apply_awgn...")
    data = np.zeros(5, dtype=complex)
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    y1 = apply_awgn(data, 10, rng1)
    y2 = apply_awgn(data, 10, rng2)
    assert np.array_equal(y1, y2), "AWGN not reproducible with same seed"
    assert y1.shape == data.shape, "Output shape mismatch"
    print("  ✓ apply_awgn passed")

def test_apply_channel():
    print("Testing apply_channel...")
    data = np.array([1+1j, 2+2j])
    H = np.array([1-1j, 0.5+0.5j])
    out = apply_channel(data, H)
    assert np.allclose(out, data * H), "apply_channel multiplication incorrect"
    try:
        apply_channel(data, H[:-1])
    except ValueError:
        print("  ✓ shape mismatch error raised")
    else:
        assert False, "apply_channel shape error not raised"
    print("  ✓ apply_channel passed")

def test_generate_rayleigh_mpth():
    print("Testing generate_rayleigh_mpth...")
    rng = np.random.default_rng(456)
    H = generate_rayleigh_mpth(100000, rng)
    power = np.mean(np.abs(H)**2)
    assert abs(power - 1) < 0.02, f"Rayleigh power off: {power}"
    print("  ✓ generate_rayleigh_mpth passed (power ≈ 1)")

def test_generate_doppler_mpth():
    print("Testing generate_doppler_mpth...")
    rng = np.random.default_rng(789)
    H = generate_doppler_mpth(10000, paths=4, speed_kmh=60, carrier_freq=1e9, rng=rng)
    assert len(H) == 10000, "Doppler output length mismatch"
    power = np.mean(np.abs(H)**2)
    assert abs(power - 1) < 0.05, f"Doppler power off: {power}"
    for invalid in [(0,4,60,1e9), (10,0,60,1e9), (10,4,60,0)]:
        try:
            generate_doppler_mpth(*invalid, rng=rng)
        except ValueError:
            pass
        else:
            assert False, "Invalid parameters not raising"
    print("  ✓ generate_doppler_mpth passed")

def test_transmit_through_channel():
    print("Testing transmit_through_channel...")
    data = np.ones(100, dtype=complex)
    rng = np.random.default_rng(321)
    rx_awgn, H_awgn = transmit_through_channel(data, 'awgn', 5, rng)
    assert np.allclose(H_awgn, 1), "AWGN H should be ones"
    assert rx_awgn.shape == data.shape
    rx_ray, H_ray = transmit_through_channel(data, 'rayleigh', 5, rng)
    assert H_ray.shape == data.shape
    rx_dop, H_dop = transmit_through_channel(data, 'doppler', 5, rng, paths=3, speed_kmh=30, carrier_freq=2e9)
    assert H_dop.shape == data.shape
    try:
        transmit_through_channel(data, 'doppler', 5, rng)
    except ValueError:
        print("  ✓ doppler missing params error raised")
    else:
        assert False, "Missing doppler params did not error"
    try:
        transmit_through_channel(data, 'invalid', 5, rng)
    except ValueError:
        print("  ✓ invalid model error raised")
    else:
        assert False, "Invalid model did not error"
    print("  ✓ transmit_through_channel passed")


#test estimator.py
def test_remove_pilot_symbols():
    print("Testing remove_pilot_symbols...")
    data = np.array([10+0j, 1+1j, 2+2j, 10+0j, 3+3j, 4+4j], dtype=complex)
    payload, pilots = remove_pilot_symbols(data, pilot_spacing=2)
    assert np.array_equal(pilots, np.array([10+0j, 10+0j])), "Pilots extracted incorrectly"
    assert np.array_equal(payload, np.array([1+1j, 2+2j, 3+3j, 4+4j])), "Payload extracted incorrectly"
    try:
        remove_pilot_symbols(data, 0)
    except ValueError:
        print("  ✓ pilot_spacing <1 error raised")
    else:
        assert False, "No error for pilot_spacing<1"
    print("  ✓ remove_pilot_symbols passed")

def test_fft_interpolate_complex():
    print("Testing fft_interpolate_complex...")
    # constant pilots => constant interpolation
    pilots = np.array([1+1j]*4)
    interp = fft_interpolate_complex(pilots, pilot_spacing=2)
    assert interp.shape[0] == 4*2, "Interpolated length mismatch"
    assert np.allclose(interp, 1+1j), "Constant pilot interpolation failed"
    # error conditions
    try:
        fft_interpolate_complex(np.array([]), 2)
    except ValueError:
        print("  ✓ empty pilot_samples error raised")
    else:
        assert False, "No error on empty pilot_samples"
    try:
        fft_interpolate_complex(pilots, 0)
    except ValueError:
        print("  ✓ pilot_spacing <1 error raised")
    else:
        assert False, "No error for pilot_spacing<1"
    print("  ✓ fft_interpolate_complex passed")

def test_interpolate_complex_points():
    print("Testing interpolate_complex_points...")
    p1, p2 = 0+0j, 2+2j
    arr = interpolate_complex_points(p1, p2, n_interp=3)
    expected = np.array([0+0j, 1+1j, 2+2j])
    assert np.allclose(arr, expected), f"Linear interpolation failed: {arr}"
    try:
        interpolate_complex_points(p1, p2, 1)
    except ValueError:
        print("  ✓ n_interp<2 error raised")
    else:
        assert False, "No error for n_interp<2"
    print("  ✓ interpolate_complex_points passed")



if __name__ == "__main__":
    test_generate_sequence_bins()
    test_modulate_sequence()
    test_add_pilot_symbols()

    test_equalize_channel()
    test_separate_real_imaginary()
    test_qam_demod_roundtrip()
    test_psk_demod_roundtrip()
    test_bit_symbol_conversion()

    test_snr_db_to_noise_sigma()
    test_apply_awgn()
    test_apply_channel()
    test_generate_rayleigh_mpth()
    test_generate_doppler_mpth()
    test_transmit_through_channel()

    test_remove_pilot_symbols()
    test_fft_interpolate_complex()
    test_interpolate_complex_points()