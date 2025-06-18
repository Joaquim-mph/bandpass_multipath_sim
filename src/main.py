import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transmitter import generate_sequence_bins, modulate_sequence, add_pilot_symbols
from channel     import apply_awgn, apply_channel, generate_rayleigh_mpth, generate_doppler_mpth
from estimator   import fft_interpolate_complex, interpolate_complex_points, remove_pilot_symbols
from receiver    import equalize_channel, demod, symbol_indices_to_bits, bits_to_symbol_indices
from metrics     import bit_error_rate


# Global storage for simulation results
results: list = []


def record_results(
    mod: int,
    pilot_spacing: int,
    channel_model: str,
    estimator: str,
    snr_db: int,
    ber: float
) -> None:
    """
    Append a BER measurement to the global results list.
    """
    results.append({
        'modulation': mod,
        'pilot_spacing': pilot_spacing,
        'channel_model': channel_model,
        'estimator': estimator,
        'snr_db': snr_db,
        'ber': ber
    })


def save_and_plot(
    output_csv: str = 'results.csv',
    output_dir: str = 'figures'
) -> None:
    """
    Save aggregated BER results to CSV and plot BER vs. SNR curves.
    """
    # Convert to DataFrame
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Group by scenario
    group_cols = ['modulation','pilot_spacing','channel_model','estimator']
    for keys, grp in df.groupby(group_cols):
        mod, P, chan, est = keys
        avg = grp.groupby('snr_db')['ber'].mean()

        plt.figure()
        plt.semilogy(avg.index, avg.values, marker='o', label=f'{est}')
        plt.xlabel('SNR (dB)')
        plt.ylabel('BER')
        plt.title(f'M={mod}, P={P}, channel={chan}, est={est}')
        plt.grid(True, which='both', linestyle='--')
        plt.legend()

        fname = f'{output_dir}/ber_M{mod}_P{P}_{chan}_{est}.png'
        plt.savefig(fname)
        plt.close()


if __name__ == '__main__':
    rng = np.random.default_rng(123)
    paths = 5
    speed = 30      # km/h
    freq = 700e6    # Hz

    # Simulation parameters
    mods = [4, 16]
    spacings = [5, 10, 20]
    models = ['awgn', 'rayleigh', 'doppler']
    snr_range = range(-2, 31)
    trials = 2

    for mod in mods:
        for P in spacings:
            for channel_model in models:
                for snr_db in snr_range:
                    for trial in range(trials):
                        # 1) raw bits
                        n_bits = 100000
                        bits = rng.integers(0, 2, size=n_bits, dtype=int) 
                        tx_bits = rng.integers(0, 2, size=n_bits, dtype=int)

                        # 2) bit→symbol mapping
                        tx_symbols = bits_to_symbol_indices(tx_bits, mod)

                        # 3) modulation & pilot insertion
                        symbols = modulate_sequence(tx_symbols, mod)
                        tx = add_pilot_symbols(symbols, mod, P)
                        
                        # symbol indices recovered
                        rx_symbols_hat = demod(sym_eq, mod, 'QAM' if mod>4 else 'PSK')

                        # 4) back to bits
                        rx_bits_hat = symbol_indices_to_bits(rx_symbols_hat, mod)

                        # 5) now compare raw bits:
                        assert len(rx_bits_hat) == len(tx_bits), (len(rx_bits_hat), len(tx_bits))
                        ber = bit_error_rate(tx_bits, rx_bits_hat)


                        # Channel + AWGN
                        if channel_model == 'awgn':
                            rx, H_true = apply_awgn(tx, snr_db, rng), np.ones_like(tx)
                        elif channel_model == 'rayleigh':
                            H_true = generate_rayleigh_mpth(len(tx), rng)
                            rx = apply_awgn(apply_channel(tx, H_true), snr_db, rng)
                        else:  # doppler
                            H_true = generate_doppler_mpth(len(tx), paths, speed, freq, rng)
                            rx = apply_awgn(apply_channel(tx, H_true), snr_db, rng)

                        # Perfect CSI
                        sym_eq = equalize_channel(rx, H_true)
                        syms_hat = demod(sym_eq, mod, 'QAM' if mod>4 else 'PSK')
                        bits_hat = symbol_indices_to_bits(syms_hat, mod)
                        print(len(bits))
                        print(len(bits_hat))
                        ber = bit_error_rate(bits, bits_hat)
                        record_results(mod, P, channel_model, 'perfect', snr_db, ber)

                        # Pilot-based estimators
                        payload, pilot_rx = remove_pilot_symbols(rx, P)
                        _, pilot_H = remove_pilot_symbols(H_true, P)

                        # FFT interp
                        H_fft = fft_interpolate_complex(pilot_H, P)
                        sym_eq_fft = equalize_channel(rx, H_fft)
                        syms_fft = demod(sym_eq_fft, mod, 'QAM' if mod>4 else 'PSK')
                        ber_fft = bit_error_rate(bits, symbol_indices_to_bits(syms_fft, mod))
                        record_results(mod, P, channel_model, 'fft', snr_db, ber_fft)

                        # Spline interp
                        H_spl = interpolate_complex_points(pilot_H, P+1, P*len(pilot_H)+1)
                        sym_eq_spl = equalize_channel(rx, H_spl)
                        syms_spl = demod(sym_eq_spl, mod, 'QAM' if mod>4 else 'PSK')
                        ber_spl = bit_error_rate(bits, symbol_indices_to_bits(syms_spl, mod))
                        record_results(mod, P, channel_model, 'spline', snr_db, ber_spl)

    save_and_plot()
