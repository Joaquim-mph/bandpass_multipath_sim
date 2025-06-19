import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from transmitter import generate_sequence_bins, modulate_sequence, add_pilot_symbols
from channel     import transmit_through_channel
from estimator   import fft_interpolate_complex, interpolate_complex_points, remove_pilot_symbols
from receiver    import equalize_channel, demod, symbol_indices_to_bits, bits_to_symbol_indices
from metrics     import bit_error_rate
from plots import plot_constellations

rng = np.random.default_rng(42)

# Test parameters (as in full_pipeline_test)
modulation_order = 16     # 16-QAM
n_bits           = 100_000
pilot_spacing    = 5
snr_db           = 10
channel_model    = 'awgn'
paths            = 5
speed_kmh        = 30
carrier_freq     = 700e6

# 1) Bits → symbols → modulation
bits       = rng.integers(0, 2, size=n_bits)
symbols    = bits_to_symbol_indices(bits, modulation_order)
tx_symbols = modulate_sequence(symbols, modulation_order)

# 2) Insert pilots
txp = add_pilot_symbols(tx_symbols, modulation_order, pilot_spacing)

# 3) Transmit through channel + AWGN
rx, H_true = transmit_through_channel(
    txp, channel_model, snr_db, rng,
    paths=paths, speed_kmh=speed_kmh, carrier_freq=carrier_freq
)

# 4) Remove pilots
payload_rx, pilots_rx     = remove_pilot_symbols(rx, pilot_spacing)
payload_Htrue, pilots_H   = remove_pilot_symbols(H_true, pilot_spacing)

# 5) Equalize
# Perfect CSI
eq_perfect = equalize_channel(payload_rx, payload_Htrue)
# FFT interpolation (truncate)
H_fft_full = fft_interpolate_complex(pilots_H, pilot_spacing)
H_fft      = H_fft_full[: payload_rx.size ]
eq_fft     = equalize_channel(payload_rx, H_fft)
# Cubic interpolation
H_cubic = []
for i in range(len(pilots_H)-1):
    seg = interpolate_complex_points(
        pilots_H[i], pilots_H[i+1], pilot_spacing+1
    )
    H_cubic.extend(seg[:-1])
H_cubic  = np.array(H_cubic, dtype=complex)
eq_cubic = equalize_channel(payload_rx, H_cubic)



# ───────────── Your pipeline code ─────────────

# Test parameters
modulation_order = 16     # 16-QAM
n_bits           = 100_000
pilot_spacing    = 5
snr_db           = 20
channel_model    = 'awgn'
paths            = 5
speed_kmh        = 30
carrier_freq     = 700e6

# 1) Bits → symbols → modulation
bits       = rng.integers(0, 2, size=n_bits)
symbols    = bits_to_symbol_indices(bits, modulation_order)
tx_symbols = modulate_sequence(symbols, modulation_order)

# 2) Insert pilots
txp = add_pilot_symbols(tx_symbols, modulation_order, pilot_spacing)

# 3) Transmit through channel + AWGN
rx, H_true = transmit_through_channel(
    txp, channel_model, snr_db, rng,
    paths=paths, speed_kmh=speed_kmh, carrier_freq=carrier_freq
)

# 4) Remove pilots
payload_rx, pilots_rx     = remove_pilot_symbols(rx, pilot_spacing)
payload_Htrue, pilots_H   = remove_pilot_symbols(H_true, pilot_spacing)

# 5) Equalize
# Perfect CSI
eq_perfect = equalize_channel(payload_rx, payload_Htrue)
# FFT interpolation (truncate)
H_fft_full = fft_interpolate_complex(pilots_H, pilot_spacing)
H_fft      = H_fft_full[: payload_rx.size ]
eq_fft     = equalize_channel(payload_rx, H_fft)
# Cubic interpolation
H_cubic = []
for i in range(len(pilots_H)-1):
    seg = interpolate_complex_points(
        pilots_H[i], pilots_H[i+1], pilot_spacing+1
    )
    H_cubic.extend(seg[:-1])
H_cubic  = np.array(H_cubic, dtype=complex)
eq_cubic = equalize_channel(payload_rx, H_cubic)

# ─── Plot the constellations ───
plot_constellations(
    tx_symbols,
    symbols,
    [eq_perfect, eq_fft, eq_cubic],
    ["Perfect CSI", "FFT Interp", "Cubic Interp"]
)
