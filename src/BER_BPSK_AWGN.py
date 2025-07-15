import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from numba import njit, prange


# 1) JIT‐compiled compute function
@njit(parallel=True)
def compute_ber_sim(
    num_bits: int,
    max_runs: int,
    Eb: float,
    SNR_lin: np.ndarray
) -> np.ndarray:
    nSNR = SNR_lin.size
    BER_sim = np.empty(nSNR, dtype=np.float64) # alocates space for BER(SNR)

    for i in prange(nSNR):
        No = Eb / SNR_lin[i]
        acc = 0.0

        for run in range(max_runs):
            # random bits
            data = np.random.randint(0, 2, num_bits)
            s    = 2 * data - 1

            # AWGN noise
            noise = np.sqrt(No / 2.0) * np.random.randn(num_bits)

            # received
            y = s + noise

            # bit errors
            # (y>0) gives boolean array; != data coerces data to bool then back to int
            err = 0
            for k in range(num_bits):
                # manual loop to avoid any fancy pandas or vector ops
                bit_hat = 1 if y[k] > 0 else 0
                if bit_hat != data[k]:
                    err += 1
            acc += err / num_bits

        BER_sim[i] = acc / max_runs

    return BER_sim


def simulate_bpsk(
    num_bits=1000000,
    max_runs=21,
    Eb=1.0,
    SNR_dB=np.arange(-10, 20.5, 0.5),
):
    # Precompute SNR linear scale
    SNR_lin = 10.0 ** (SNR_dB / 10.0)

    # First call will compile; subsequent calls are fast
    BER_sim = compute_ber_sim(num_bits, max_runs, Eb, SNR_lin)

    # Theoretical BER
    BER_th = 0.5 * erfc(np.sqrt(SNR_lin))

    # Plotting
    plt.figure(figsize=(8,5))
    plt.semilogy(SNR_dB, BER_th, 'k-',   label='Theoretical')
    plt.semilogy(SNR_dB, BER_sim, 'k*',  label='Simulation')
    plt.grid(True)
    #plt.grid(True, which='both', ls='--', lw=0.5)
    plt.xlabel('Eb/No (dB)')
    plt.ylabel('Bit Error Rate')
    plt.title('BPSK AWGN')
    plt.legend()
    #plt.ylim(1e-7, 1)
    plt.savefig("figures/BER_BPSK_AWGN.pdf", dpi = 300)
  
    
    return SNR_dB, BER_th, BER_sim

if __name__ == "__main__":
    simulate_bpsk()