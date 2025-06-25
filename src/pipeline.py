import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from itertools import product
from transmitter import generate_sequence_bins, modulate_sequence, add_pilot_symbols
from channel     import transmit_through_channel
from estimator   import fft_interpolate_complex, interpolate_complex_points
from receiver    import equalize_channel, demod, symbol_indices_to_bits, bits_to_symbol_indices, remove_pilot_symbols
from metrics     import bit_error_rate, theoretical_ber_awgn, theoretical_ber_rayleigh
from plots import plot_constellations, compare_snr_constellations, plot_doppler_swirl
from utils import bits_per_symbol


rng = np.random.default_rng(42)

# 1) Define parameter lists
modulations    = [4, 16]
pilot_spacings = [5]
snr_db_list    = list(range(-2, 31))
channel_models = ['awgn', 'rayleigh', 'doppler']
doppler_params = {
    'paths': [5, 40],
    'speed_kmh': [30, 120],
    'carrier_freq': [700e6, 3.5e9]
}


# ───────────── Your pipeline code ─────────────

# Test parameters
modulation_order = 4    
n_bits           = 100_000
pilot_spacing    = 5
snr_db           = -5
channel_model    = 'doppler'
paths            = 5
speed_kmh        = 30
carrier_freq     = 700e6

# 1) Bits → symbols → modulation
# Directly generate n_bits / bps random symbol indices:
symbols    = generate_sequence_bins(modulation_order, n_bits)
tx_symbols = modulate_sequence(symbols, modulation_order)
bits = symbol_indices_to_bits(symbols, modulation_order)

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

# 6) Demodulate each equalized stream back to symbol indices
rec_p = demod(eq_perfect, modulation_order, 'PSK')
rec_f = demod(eq_fft,     modulation_order, 'PSK')
rec_c = demod(eq_cubic,   modulation_order, 'PSK')

rec_p = np.array(rec_p, dtype=int)
rec_f = np.array(rec_f, dtype=int)
rec_c = np.array(rec_c, dtype=int)

# 7) Convert symbol indices to bit-streams
bits_p = symbol_indices_to_bits(rec_p, modulation_order)
bits_f = symbol_indices_to_bits(rec_f, modulation_order)
bits_c = symbol_indices_to_bits(rec_c, modulation_order)

# 8) Compute empirical BER
ber_p = bit_error_rate(bits, bits_p)
ber_f = bit_error_rate(bits, bits_f)
ber_c = bit_error_rate(bits, bits_c)

print(f"Empirical BER (perfect CSI): {ber_p:.3e}")
print(f"Empirical BER (FFT CSI)   : {ber_f:.3e}")
print(f"Empirical BER (Cubic CSI) : {ber_c:.3e}")

# 9) Compute theoretical BER for AWGN & Rayleigh at the same Eb/N0
#    Note: your snr_db is Es/N0.  For PSK Q=2 bits/symbol:
bps     = bits_per_symbol(modulation_order)        # 2 for QPSK
ebno_db = snr_db - 10*math.log10(bps)              # Eb/N0 = Es/N0 – 10log10(bps)

ber_awgn    = theoretical_ber_awgn(modulation_order, ebno_db)
ber_rayleigh= theoretical_ber_rayleigh(modulation_order, ebno_db)

print(f"Theoretical BER AWGN     : {ber_awgn:.3e}")
print(f"Theoretical BER Rayleigh : {ber_rayleigh:.3e}")



# plot_constellations(
#     tx_symbols,
#     symbols,
#     [eq_perfect, eq_fft, eq_cubic],
#     ["Perfect CSI", "FFT Interp", "Cubic Interp"]
# )


# compare_snr_constellations(4, 'doppler', snr_list=[-5,0,10,30], n_bits=100_000)
# plot_doppler_swirl(16, 'doppler', snr_list=[-5, 0, 10, 30], n_bits=100000)