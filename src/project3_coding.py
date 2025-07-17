import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from transmitter import generate_sequence_bins, modulate_sequence, add_pilot_symbols
from channel import transmit_through_channel
from utils import fft_interpolate_complex, interpolate_complex_points, bit_error_rate, bits_per_symbol
from receiver import equalize_channel, demod, symbol_indices_to_bits, bits_to_symbol_indices, remove_pilot_symbols
from concurrent.futures import ProcessPoolExecutor
from joblib import Parallel, delayed
from numba import cuda, uint8

# import BCH encode/decode kernels
from BCH_CUDA import encode_stream_gpu, decode_stream_gpu

# BCH parameters
n = 7
k = 4
R = k/n
snr_coded_factor = 10*np.log10(R)
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
n_bits = 100_000    # bits per run
runs   = 21         # number of Monte Carlo runs per scenario

# 3) Container for results
rows = []
shift_4 = 10*np.log10(4 * R)
shift_16 = 10*np.log10(16 * R)

def single_run(seed, M, P, snr_db, model, scenario_tag, ch_args, n_bits):
    if M == 4:
        snr_db += shift_4
    else:
        snr_db += shift_16

    rng_run = np.random.default_rng(seed)
    # generate random bit payload
    bits = rng_run.integers(0, 2, size=n_bits).astype(np.uint8)
    blocks = n_bits // k

    # BCH encode on GPU
    msg_dev = cuda.to_device(bits)
    cw_dev = cuda.device_array(blocks * n, dtype=np.uint8)
    threads = 512
    grid = (blocks + threads - 1) // threads
    encode_stream_gpu[grid, threads](msg_dev, cw_dev)
    cw_host = cw_dev.copy_to_host()

    # map coded bits to symbols
    bps = int(np.log2(M))
    syms = bits_to_symbol_indices(cw_host, bps)
    tx = modulate_sequence(syms, M)
    txp = add_pilot_symbols(tx, M, P)

    # transmit through channel
    rx, H = transmit_through_channel(txp, model, snr_db, rng_run, **ch_args)

    # remove pilots and equalize
    payload_rx, pilots_rx = remove_pilot_symbols(rx, P)
    payload_H, pilots_H = remove_pilot_symbols(H, P)

    eq_perf = equalize_channel(payload_rx, payload_H)
    # cubic interpolation
    H_cubic = []
    for i in range(len(pilots_H) - 1):
        seg = interpolate_complex_points(pilots_H[i], pilots_H[i + 1], P + 1)
        H_cubic.extend(seg[:-1])
    eq_cub = equalize_channel(payload_rx, np.array(H_cubic, dtype=complex))

    # demod to coded bit indices
    rec_p = np.array(demod(eq_perf, M, 'QAM' if M > 4 else 'PSK'), dtype=np.int32)
    rec_c = np.array(demod(eq_cub, M, 'QAM' if M > 4 else 'PSK'), dtype=np.int32)

    # indices to bits
    cw_rec_p = symbol_indices_to_bits(rec_p, bps)
    cw_rec_c = symbol_indices_to_bits(rec_c, bps)

    # BCH decode on GPU
    cw_dev_p = cuda.to_device(cw_rec_p)
    msg_dev_out_p = cuda.device_array(blocks * k, dtype=np.uint8)
    decode_stream_gpu[grid, threads](cw_dev_p, msg_dev_out_p)
    dec_p = msg_dev_out_p.copy_to_host()

    cw_dev_c = cuda.to_device(cw_rec_c)
    msg_dev_out_c = cuda.device_array(blocks * k, dtype=np.uint8)
    decode_stream_gpu[grid, threads](cw_dev_c, msg_dev_out_c)
    dec_c = msg_dev_out_c.copy_to_host()

    # compute BER
    ber_p = np.mean(dec_p != bits)
    ber_c = np.mean(dec_c != bits)

    return {'perfect': ber_p, 'cubic': ber_c}



if __name__ == "__main__":
    for M, P, snr_db in product(modulations, pilot_spacings, snr_db_list):
        # assemble scenarios
        scenarios = []
        for model in channel_models:
            if model == 'doppler':
                for paths, speed, freq in product(
                    doppler_params['paths'], doppler_params['speed_kmh'], doppler_params['carrier_freq']
                ):
                    tag = f"doppler_p{paths}_v{speed}_f{int(freq/1e6)}MHz"
                    scenarios.append((model, tag, {'paths': paths, 'speed_kmh': speed, 'carrier_freq': freq}))
            else:
                scenarios.append((model, model, {}))
        for model, scenario_tag, ch_args in scenarios:
            seeds = [int.from_bytes(os.urandom(4), 'little') for _ in range(runs)]
            ber_list = Parallel(n_jobs=-1)(
                delayed(single_run)(seed, M, P, snr_db, model, scenario_tag, ch_args, n_bits)
                for seed in seeds
            )
            # average results\            
            sum_ber = {'perfect': 0.0, 'cubic': 0.0}
            for run in ber_list:
                sum_ber['perfect'] += run['perfect']
                sum_ber['cubic'] += run['cubic']
            for method in sum_ber:
                rows.append({
                    'modulation': M,
                    'pilot_spacing': P,
                    'snr_db': snr_db,
                    'channel': model,
                    'scenario': scenario_tag,
                    'method': method,
                    'ber': sum_ber[method] / runs
                })
    df = pd.DataFrame(rows)
    df.to_csv('BER_sweep_results_bch.csv', index=False)
    print("Sweep complete — results in 'BER_sweep_results_bch.csv'")
