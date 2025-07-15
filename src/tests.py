import math
import numpy as np
import pytest

from transmitter import (
    generate_sequence_bins,
    modulate_sequence,
    add_pilot_symbols
)
from receiver import (
    equalize_channel,
    separate_real_imaginary,
    demod,
    symbol_indices_to_bits,
    bits_to_symbol_indices,
    remove_pilot_symbols
)
from channel import (
    apply_awgn,
    apply_channel,
    generate_rayleigh_mpth,
    generate_doppler_mpth,
    transmit_through_channel
)
from utils import (
    PILOT,
    bits_per_symbol,
    get_pilot_symbol,
    snr_db_to_noise_sigma,
    get_normalized_constellation
)


def test_constellation_normalization():
    """
    Ensure every normalized constellation has unit average energy.
    """
    for M in [4, 16]:
        lut = get_normalized_constellation(M)
        avg_energy = np.mean(np.abs(lut)**2)
        assert math.isclose(avg_energy, 1.0, rel_tol=1e-6), \
            f"M={M} constellation not normalized (avg_energy={avg_energy:.6f})"
    # also test that invalid M raises
    with pytest.raises(ValueError):
        _ = get_normalized_constellation(8)

# -------------------------------------------------------------------------
# Transmitter tests
# -------------------------------------------------------------------------

def test_generate_sequence_bins():
    for mod in [4, 16]:
        for n_bits in [16, 32]:
            seq = generate_sequence_bins(mod, n_bits)
            expected_len = n_bits // bits_per_symbol(mod)
            assert len(seq) == expected_len
            assert seq.min() >= 0 and seq.max() < mod


def test_modulate_sequence():
    seq = np.array([0, 1, 2, 3])
    tx = modulate_sequence(seq, 4)
    avg_energy = np.mean(np.abs(tx)**2)
    assert math.isclose(avg_energy, 1.0, rel_tol=1e-6)


def test_add_pilot_symbols():
    data = np.array([1+0j, 2+0j, 3+0j, 4+0j])
    out = add_pilot_symbols(data, 4, 2)
    pilot = get_pilot_symbol(4)
    expected = np.array([pilot, 1+0j, 2+0j, pilot, 3+0j, 4+0j, pilot], dtype=complex)
    assert np.allclose(out, expected)

# -------------------------------------------------------------------------
# Receiver tests
# -------------------------------------------------------------------------

def test_equalize_channel():
    data = np.array([1+2j, 2+4j, 3+6j])
    H = np.array([1+1j, 2+0j, 3-3j])
    eq = equalize_channel(data, H)
    assert np.allclose(eq, data / H)
    with pytest.raises(ValueError):
        equalize_channel(data, H[:-1])


def test_separate_real_imaginary():
    data = np.array([1+2j, -1-0.5j, 0+1j])
    I, Q = separate_real_imaginary(data)
    assert np.array_equal(I, [1, -1, 0])
    assert np.array_equal(Q, [2, -0.5, 1])


def test_demod_roundtrip():
    # QAM
    sym_qam = np.random.randint(0, 16, size=10)
    tx_qam = modulate_sequence(sym_qam, 16)
    rec_qam = demod(tx_qam, 16, 'QAM')
    assert rec_qam == sym_qam.tolist()
    # PSK
    sym_psk = np.random.randint(0, 4, size=10)
    tx_psk = modulate_sequence(sym_psk, 4)
    rec_psk = demod(tx_psk, 4, 'PSK')
    assert rec_psk == sym_psk.tolist()


def test_bit_symbol_conversion():
    bits = np.random.randint(0, 2, size=32)
    bps = bits_per_symbol(16)
    syms = bits_to_symbol_indices(bits, bps)
    bits_rt = symbol_indices_to_bits(syms, bps)
    assert bits_rt.tolist() == bits.tolist()


def test_remove_pilot_symbols():
    data = np.array([10+0j, 1+1j, 2+2j, 10+0j, 3+3j, 4+4j], dtype=complex)
    payload, pilots = remove_pilot_symbols(data, pilot_spacing=2)
    assert np.array_equal(pilots, [10+0j, 10+0j])
    assert np.array_equal(payload, [1+1j, 2+2j, 3+3j, 4+4j])
    with pytest.raises(ValueError):
        remove_pilot_symbols(data, 0)

# -------------------------------------------------------------------------
# Channel tests
# -------------------------------------------------------------------------

def test_snr_db_to_noise_sigma():
    assert math.isclose(snr_db_to_noise_sigma(0), math.sqrt(1/2), rel_tol=1e-6)
    assert math.isclose(snr_db_to_noise_sigma(3), math.sqrt(0.5 / 10**(3/10)), rel_tol=1e-6)


def test_apply_awgn():
    data = np.zeros(5, dtype=complex)
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    y1 = apply_awgn(data, 10, rng1)
    y2 = apply_awgn(data, 10, rng2)
    assert np.array_equal(y1, y2)
    assert y1.shape == data.shape


def test_apply_channel():
    data = np.array([1+1j, 2+2j])
    H = np.array([1-1j, 0.5+0.5j])
    out = apply_channel(data, H)
    assert np.allclose(out, data * H)
    with pytest.raises(ValueError):
        apply_channel(data, H[:-1])


def test_generate_rayleigh_mpth():
    rng = np.random.default_rng(456)
    H = generate_rayleigh_mpth(100000, rng)
    power = np.mean(np.abs(H)**2)
    assert abs(power - 1) < 0.02


def test_generate_doppler_mpth():
    rng = np.random.default_rng(789)
    H = generate_doppler_mpth(10000, paths=4, speed_kmh=60, carrier_freq=1e9, rng=rng)
    assert len(H) == 10000
    power = np.mean(np.abs(H)**2)
    assert abs(power - 1) < 0.05
    with pytest.raises(ValueError):
        generate_doppler_mpth(0,4,60,1e9, rng)
    with pytest.raises(ValueError):
        generate_doppler_mpth(10,0,60,1e9, rng)
    with pytest.raises(ValueError):
        generate_doppler_mpth(10,4,60,0, rng)


def test_transmit_through_channel():
    data = np.ones(100, dtype=complex)
    rng = np.random.default_rng(321)
    rx_awgn, H_awgn = transmit_through_channel(data, 'awgn', 5, rng)
    assert np.allclose(H_awgn, np.ones_like(data))
    assert rx_awgn.shape == data.shape
    rx_ray, H_ray = transmit_through_channel(data, 'rayleigh', 5, rng)
    assert H_ray.shape == data.shape
    rx_dop, H_dop = transmit_through_channel(data, 'doppler', 5, rng, paths=3, speed_kmh=30, carrier_freq=2e9)
    assert H_dop.shape == data.shape
    with pytest.raises(ValueError):
        transmit_through_channel(data, 'doppler', 5, rng)
    with pytest.raises(ValueError):
        transmit_through_channel(data, 'invalid', 5, rng)




if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__]))
