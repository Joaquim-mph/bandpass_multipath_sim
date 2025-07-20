import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from transmitter import generate_sequence_bins, modulate_sequence, add_pilot_symbols
from channel     import transmit_through_channel
from utils       import fft_interpolate_complex, interpolate_complex_points, bit_error_rate, bits_per_symbol, ebn0_to_snr_db
from receiver    import equalize_channel, demod, symbol_indices_to_bits, bits_to_symbol_indices, remove_pilot_symbols
from concurrent.futures import ProcessPoolExecutor
from joblib      import Parallel, delayed


# 1) Define parameter lists
modulations    = [4, 16]
pilot_spacings = [5, 10]
snr_db_list    = list(range(-2, 31))  # –2 to 30 dB
channel_models = ['doppler']
doppler_params = {
    'paths': [10],
    'speed_kmh': [50],
    'carrier_freq': [700e6, 3.5e9]
}

# 2) Simulation settings
n_bits = 100_000     # bits per run
runs   = 21         # number of Monte Carlo runs per scenario

# 3) Container for results
rows = []


# 3) Fixed random generator for reproducibility
rng = np.random.default_rng(1111)


# 4) Define a helper to do pilot-removal, equalize, demod & BER
def process_one(rx, H, bits, M, P, snr_db, model, scenario_tag):
    payload_rx, pilots_rx   = remove_pilot_symbols(rx, P)
    payload_H,  pilots_H    = remove_pilot_symbols(H,  P)

    # Perfect CSI
    eq_perfect = equalize_channel(payload_rx, payload_H)

    # Cubic interp
    H_cubic = []
    for i in range(len(pilots_H)-1):
        seg = interpolate_complex_points(pilots_H[i], pilots_H[i+1], P+1)
        H_cubic.extend(seg[:-1])
    H_cubic = np.array(H_cubic, dtype=complex)
    eq_cubic = equalize_channel(payload_rx, H_cubic)

    # Demod & bits
    rec_p = demod(eq_perfect, M, 'QAM' if M>4 else 'PSK')
    rec_c = demod(eq_cubic,   M, 'QAM' if M>4 else 'PSK')

    rec_p = np.array(rec_p, dtype=int)
    rec_c = np.array(rec_c, dtype=int)

    bits_p = symbol_indices_to_bits(rec_p, M)
    bits_c = symbol_indices_to_bits(rec_c, M)

    # Compute BERs
    ber_p = np.mean(np.array(bits_p) != bits)
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
        ('cubic'  , ber_c),
    ]]

# 4) Helper for one Monte Carlo run
def single_run(seed, M, P, snr_db, model, scenario_tag, ch_args):
    rng_run = np.random.default_rng(seed)
    syms = generate_sequence_bins(M, n_bits)
    bits = symbol_indices_to_bits(syms, M)
    tx   = modulate_sequence(syms, M)
    txp  = add_pilot_symbols(tx, M, P)
    bps    = bits_per_symbol(M)                   # 2 for QPSK, 4 for 16-QAM
    esn0_db = ebn0_to_snr_db(snr_db, bps)           # add 10·log10(bps)
    rx, H = transmit_through_channel(txp, model, esn0_db, rng_run, **ch_args)
    results = process_one(rx, H, bits, M, P, snr_db, model, scenario_tag)
    # extract per-method BER for this run
    return {res['method']: res['ber'] for res in results}



if __name__ == "__main__":

    # 5) Sweep loops
    for M, P, snr_db in product(modulations, pilot_spacings, snr_db_list):
        # assemble scenarios
        scenarios = []
        for model in channel_models:
            if model == 'doppler':
                for paths, speed, freq in product(
                    doppler_params['paths'],
                    doppler_params['speed_kmh'],
                    doppler_params['carrier_freq']
                ):
                    tag = f"doppler_p{paths}_v{speed}_f{int(freq/1e6)}MHz"
                    scenarios.append((model, tag, {'paths':paths, 'speed_kmh':speed, 'carrier_freq':freq}))
            else:
                scenarios.append((model, model, {}))
        # for each scenario, parallelize runs
        for model, scenario_tag, ch_args in scenarios:
            # generate distinct seeds
            seeds = [int.from_bytes(os.urandom(4), 'little') for _ in range(runs)]
            # run Monte Carlo in parallel
            ber_list = Parallel(n_jobs=-1)(
                delayed(single_run)(seed, M, P, snr_db, model, scenario_tag, ch_args)
                for seed in seeds
            )
            # sum and average
            sum_ber = {'perfect':0.0, 'cubic':0.0}
            for run_ber in ber_list:
                for method in sum_ber:
                    sum_ber[method] += run_ber[method]
            for method, total in sum_ber.items():
                rows.append({
                    'modulation':    M,
                    'pilot_spacing': P,
                    'snr_db':        snr_db,
                    'channel':       model,
                    'scenario':      scenario_tag,
                    'method':        method,
                    'ber':           total / runs
                })

    # 6) Save results
    df = pd.DataFrame(rows)
    df.to_csv('BER_sweep_results.csv', index=False)
    print("Sweep complete — results in 'BER_sweep_results.csv'")