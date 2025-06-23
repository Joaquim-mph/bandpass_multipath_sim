import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Sequence
from utils import QPSK, QAM16
from transmitter import generate_sequence_bins, modulate_sequence, add_pilot_symbols
from receiver import remove_pilot_symbols, equalize_channel
from channel import transmit_through_channel


def _build_constellation_lut(M: int):
    if M == 4:
        norm = np.sqrt(2)
        lut = QPSK
    else:
        norm = np.sqrt(2 * (M - 1) / 3)
        lut = QAM16
    return np.array([lut[s] / norm for s in range(M)], dtype=complex)

def plot_constellations(
    tx: np.ndarray,
    sym_indices: Sequence[int],
    eqs: List[np.ndarray],
    labels: List[str],
):
    """
    Scatter‐plot the ideal TX constellation and equalized RX constellations,
    coloring *each point* by its original symbol index so you can track it.
    """
    # Number of symbols
    M = len(np.unique(sym_indices))
    bps = int(np.log2(M))
    cmap = plt.cm.get_cmap('tab20', M)
    norm = mcolors.BoundaryNorm(boundaries=np.arange(M+1)-0.5, ncolors=M)

    # Build LUT once (for nearest‐neighbor if you ever need it)
    constellation = _build_constellation_lut(M)

    # Prepare figure
    n_plots = 1 + len(eqs)
    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))

    # Convert sym_indices to array
    idx_seg = np.array(sym_indices)

    # TX Ideal (colors = original indices)
    axes[0].scatter(
        tx.real, tx.imag,
        c=idx_seg, cmap=cmap, norm=norm, s=2
    )
    axes[0].set_title("TX Ideal")
    axes[0].set_xlabel("In‐phase"); axes[0].set_ylabel("Quadrature")
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes[0], ticks=np.arange(M)
    )
    cbar.ax.set_yticklabels([f"{i:0{bps}b}" for i in range(M)])

    # RX after equalization (reuse the same colors)
    for ax, eq, label in zip(axes[1:], eqs, labels):
        # Might have dropped pilots, so slice idx_seg to match length
        idx_eq = idx_seg[:len(eq)]
        ax.scatter(
            eq.real, eq.imag,
            c=idx_eq, cmap=cmap, norm=norm, s=2
        )
        ax.set_title(label)
        ax.set_xlabel("In‐phase"); ax.set_ylabel("Quadrature")

    plt.tight_layout()
    plt.show()



def compare_snr_constellations(
    modulation_order: int,
    model: str,
    snr_list: list = [-5, 0, 10, 30],
    n_bits: int = 2000,
    pilot_spacing: int = 5,
    paths: int = 5,
    speed_kmh: float = 30,
    carrier_freq: float = 700e6,
    rng: np.random.Generator = None
):
    """
    Compare equalized constellations across multiple SNRs in a 2×2 grid.

    Parameters
    ----------
    modulation_order
        Modulation order (4 for QPSK, 16 for 16-QAM).
    model
        Channel model ('awgn', 'rayleigh', or 'doppler').
    snr_list
        List of 4 SNR values (dB) to plot in reading order.
    n_bits
        Number of bits to generate (symbols = n_bits/log2(M)).
    pilot_spacing
        Pilots inserted every (pilot_spacing+1) symbols.
    paths, speed_kmh, carrier_freq
        Doppler parameters, if model='doppler'.
    rng
        NumPy random Generator (optional).
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1) Generate & modulate one block
    symbols = generate_sequence_bins(modulation_order, n_bits)
    tx = modulate_sequence(symbols, modulation_order)
    txp = add_pilot_symbols(tx, modulation_order, pilot_spacing)

    # Prepare color mapping by original symbol index
    M = modulation_order
    cmap = plt.cm.get_cmap('tab10', M)
    norm = mcolors.BoundaryNorm(np.arange(M+1)-0.5, M)

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, snr_db in zip(axes, snr_list):
        # 2) Transmit through channel + AWGN
        rx, H = transmit_through_channel(
            txp, model, snr_db, rng,
            paths=paths, speed_kmh=speed_kmh, carrier_freq=carrier_freq
        )
        # 3) Remove pilots
        payload_rx, pilots_rx = remove_pilot_symbols(rx, pilot_spacing)
        payload_H, _ = remove_pilot_symbols(H, pilot_spacing)
        # 4) Equalize (perfect CSI)
        eq = equalize_channel(payload_rx, payload_H)

        # 5) Scatter plot, color by original symbol index
        idx = symbols[:len(eq)]
        ax.scatter(eq.real, eq.imag, c=idx, cmap=cmap, norm=norm, s=10)
        ax.set_title(f"SNR = {snr_db} dB")
        ax.set_xlabel("ℜ(-)")
        ax.set_ylabel("ℑ(-)")
        ax.axhline(0, color='grey', lw=0.5)
        ax.axvline(0, color='grey', lw=0.5)

    fig.suptitle(f"{modulation_order}-ary {model.title()} Channel + AWGN", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



def plot_doppler_swirl(
    modulation_order: int,
    model: str,
    snr_list: list = [-5, 0, 10, 30],
    n_bits: int = 2000,
    pilot_spacing: int = 5,
    paths: int = 5,
    speed_kmh: float = 30,
    carrier_freq: float = 700e6,
    rng: np.random.Generator = None
):
    """
    Plot raw (unequalized) Doppler swirls at different SNRs in a 2×2 grid,
    coloring each point by its original symbol index.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # 1) Generate & modulate symbols
    symbols = generate_sequence_bins(modulation_order, n_bits)
    tx      = modulate_sequence(symbols, modulation_order)
    txp     = add_pilot_symbols(tx, modulation_order, pilot_spacing)

    # Prepare colormap (reuse same palette logic)
    M    = modulation_order
    cmap = plt.cm.get_cmap('tab20', M)             # or 'tab10' if you prefer
    norm = mcolors.BoundaryNorm(np.arange(M+1)-0.5, M)

    # 2×2 figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, snr_db in zip(axes, snr_list):
        # Transmit through Doppler+AWGN
        rx, H = transmit_through_channel(
            txp, model, snr_db, rng,
            paths=paths, speed_kmh=speed_kmh, carrier_freq=carrier_freq
        )

        # Remove pilots
        payload_rx, _ = remove_pilot_symbols(rx, pilot_spacing)

        # Color each point by its original symbol index
        idx = symbols[:len(payload_rx)]

        ax.scatter(
            payload_rx.real, payload_rx.imag,
            c=idx, cmap=cmap, norm=norm,
            s=10, alpha=0.7
        )
        ax.set_title(f"SNR = {snr_db} dB")
        ax.set_xlabel("ℜ(-)"); ax.set_ylabel("ℑ(-)")
        ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
        ax.set_aspect('equal')

    fig.suptitle(f"{modulation_order}-ary {model.title()} Channel Swirls (unequalized)", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()