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


# 1) Define parameter lists
modulations    = [4, 16]
pilot_spacings = [5]
snr_db_list    = list(range(-2, 31))      # –2:1:30 dB :contentReference[oaicite:1]{index=1}
channel_models = ['awgn', 'rayleigh', 'doppler']
doppler_params = {
    'paths': [5, 40],
    'speed_kmh': [30, 120],
    'carrier_freq': [700e6, 3.5e9]
}

# 2) Prepare an empty list to accumulate results
rows = []

# 3) Fixed random generator for reproducibility
rng = np.random.default_rng(1234)


# 4) Define a helper to do pilot-removal, equalize, demod & BER
def process_one(rx, H, bits, M, P, snr_db, model, scenario_tag):
    payload_rx, pilots_rx   = remove_pilot_symbols(rx, P)
    payload_H,  pilots_H    = remove_pilot_symbols(H,  P)

    # Perfect CSI
    eq_perfect = equalize_channel(payload_rx, payload_H)

    # FFT interp (then truncate to payload length)
    H_fft_full = fft_interpolate_complex(pilots_H, P)
    H_fft      = H_fft_full[: payload_rx.size ]
    eq_fft     = equalize_channel(payload_rx, H_fft)

    # Cubic interp
    H_cubic = []
    for i in range(len(pilots_H)-1):
        seg = interpolate_complex_points(pilots_H[i], pilots_H[i+1], P+1)
        H_cubic.extend(seg[:-1])
    H_cubic = np.array(H_cubic, dtype=complex)
    eq_cubic = equalize_channel(payload_rx, H_cubic)

    # Demod & bits
    rec_p = demod(eq_perfect, M, 'QAM' if M>4 else 'PSK')
    rec_f = demod(eq_fft,     M, 'QAM' if M>4 else 'PSK')
    rec_c = demod(eq_cubic,   M, 'QAM' if M>4 else 'PSK')

    bits_p = symbol_indices_to_bits(rec_p, M)
    bits_f = symbol_indices_to_bits(rec_f, M)
    bits_c = symbol_indices_to_bits(rec_c, M)

    # Compute BERs
    ber_p = np.mean(np.array(bits_p) != bits)
    ber_f = np.mean(np.array(bits_f) != bits)
    ber_c = np.mean(np.array(bits_c) != bits)

    return [{
        'modulation'   : M,
        'pilot_spacing': P,
        'snr_db'       : snr_db,
        'channel'      : model,
        'scenario'     : scenario_tag,
        'method'       : method,
        'ber'          : ber
    } for method, ber in [
        ('perfect', ber_p),
        ('fft'    , ber_f),
        ('cubic'  , ber_c),
    ]]


# 5) Loop through everything
for M, P, snr_db in product(modulations, pilot_spacings, snr_db_list):
    # a) Generate & modulate one block of bits
    bits = rng.integers(0, 2, size=2000)                  
    syms = bits_to_symbol_indices(bits, M)
    tx   = modulate_sequence(syms, M)
    txp  = add_pilot_symbols(tx, M, P)

    for model in channel_models:
        # set up arguments for transmit_through_channel
        kwargs = dict(data=txp, model=model, snr_db=snr_db, rng=rng)
        if model == 'doppler':
            # expand doppler scenarios
            for paths, speed, freq in product(
                doppler_params['paths'],
                doppler_params['speed_kmh'],
                doppler_params['carrier_freq']
            ):  
                        # unpack & equalize exactly like in process_one
                payload_rx, pilots_rx = remove_pilot_symbols(rx, P)
                payload_H,  pilots_H  = remove_pilot_symbols(H,   P)

                eq_perfect = equalize_channel(payload_rx, payload_H)

                H_fft_full = fft_interpolate_complex(pilots_H, P)
                H_fft      = H_fft_full[: payload_rx.size ]
                eq_fft     = equalize_channel(payload_rx, H_fft)

                H_cubic = []
                for i in range(len(pilots_H)-1):
                    seg = interpolate_complex_points(pilots_H[i], pilots_H[i+1], P+1)
                    H_cubic.extend(seg[:-1])
                H_cubic = np.array(H_cubic, dtype=complex)
                eq_cubic = equalize_channel(payload_rx, H_cubic)

                rx, H = transmit_through_channel(
                    **kwargs, paths=paths,
                    speed_kmh=speed,
                    carrier_freq=freq
                )
                scenario_tag = f"doppler_p{paths}_v{speed}_f{int(freq/1e6)}MHz"
                # process one scenario…
                # (see step 5–7 below)
                rows += process_one(rx, H, bits, M, P, snr_db, model, scenario_tag)

        else:
            # AWGN or Rayleigh (no extra params)
            rx, H = transmit_through_channel(**kwargs)
            rows += process_one(rx, H, bits, M, P, snr_db, model, model)
                # ─── PLOT constellations for just one case ───
        if M == 16 and P == 5 and snr_db == 10 and model == 'awgn':
            # tx are the original transmitted symbols (no pilots)
            plot_constellations(
                tx, 
                [eq_perfect, eq_fft, eq_cubic], 
                ["Perfect CSI", "FFT Interp", "Cubic Interp"],
                num_pts=2000
            )

# 6) Build DataFrame & save to CSV
df = pd.DataFrame(rows)
df.to_csv("BER_sweep_results.csv", index=False)
print("Sweep complete: results in BER_sweep_results.csv")