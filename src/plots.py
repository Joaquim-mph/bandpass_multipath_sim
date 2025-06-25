import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Sequence
from utils import QPSK, QAM16
from transmitter import generate_sequence_bins, modulate_sequence, add_pilot_symbols
from receiver import remove_pilot_symbols, equalize_channel
from channel import transmit_through_channel
import os

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



def plot_unequalized(
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
    if rng is None:
        rng = np.random.default_rng(42)

    symbols = generate_sequence_bins(modulation_order, n_bits)
    tx      = modulate_sequence(symbols, modulation_order)
    txp     = add_pilot_symbols(tx, modulation_order, pilot_spacing)

    M    = modulation_order
    cmap = plt.cm.get_cmap('tab20', M)
    norm = mcolors.BoundaryNorm(np.arange(M+1)-0.5, M)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, snr_db in zip(axes, snr_list):
        rx, _ = transmit_through_channel(
            txp, model, snr_db, rng,
            paths=paths, speed_kmh=speed_kmh, carrier_freq=carrier_freq
        )
        payload_rx, _ = remove_pilot_symbols(rx, pilot_spacing)
        idx = symbols[:len(payload_rx)]
        ax.scatter(
            payload_rx.real, payload_rx.imag,
            c=idx, cmap=cmap, norm=norm,
            s=10, alpha=0.7
        )
        ax.set_title(f"SNR = {snr_db} dB")
        ax.set_xlabel("ℜ{·}"); ax.set_ylabel("ℑ{·}")
        ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
        ax.set_aspect('equal', 'box')

    fig.suptitle(
        f"{modulation_order}-ary {model.title()} Swirls (unequalized)",
        fontsize=16
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 📁 nombre dinámico que incluye todos los parámetros relevantes
    out_dir = "constellations"
    os.makedirs(out_dir, exist_ok=True)
    if model.lower() == 'doppler':
        fname = (f"{M}_doppler_p{paths}_v{speed_kmh:.0f}kmh_"
                 f"f{int(carrier_freq/1e6)}MHz_P{pilot_spacing}_unequalized.png")
    else:
        fname = f"{M}_{model}_P{pilot_spacing}_unequalized.png"

    outpath = os.path.join(out_dir, fname)
    fig.savefig(outpath, dpi = 100)
    plt.close(fig)
    print(f"Saved unequalized plot to {outpath}")


def plot_equalized(
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
    if rng is None:
        rng = np.random.default_rng(42)

    symbols = generate_sequence_bins(modulation_order, n_bits)
    tx      = modulate_sequence(symbols, modulation_order)
    txp     = add_pilot_symbols(tx, modulation_order, pilot_spacing)

    M    = modulation_order
    cmap = plt.cm.get_cmap('tab20', M)
    norm = mcolors.BoundaryNorm(np.arange(M+1)-0.5, M)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, snr_db in zip(axes, snr_list):
        rx, H = transmit_through_channel(
            txp, model, snr_db, rng,
            paths=paths, speed_kmh=speed_kmh, carrier_freq=carrier_freq
        )
        payload_rx, pilots_rx = remove_pilot_symbols(rx, pilot_spacing)
        payload_H,  _         = remove_pilot_symbols(H,  pilot_spacing)
        eq = equalize_channel(payload_rx, payload_H)

        idx = symbols[:len(eq)]
        ax.scatter(
            eq.real, eq.imag,
            c=idx, cmap=cmap, norm=norm,
            s=10, alpha=0.8
        )
        ax.set_title(f"SNR = {snr_db} dB")
        ax.set_xlabel("ℜ{·}"); ax.set_ylabel("ℑ{·}")
        ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
        ax.set_aspect('equal', 'box')

    fig.suptitle(
        f"{modulation_order}-ary {model.title()} (equalized)",
        fontsize=16
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 📁 nombre dinámico con todos los parámetros
    out_dir = "constellations"
    os.makedirs(out_dir, exist_ok=True)
    if model.lower() == 'doppler':
        fname = (f"{M}_doppler_p{paths}_v{speed_kmh:.0f}kmh_"
                 f"f{int(carrier_freq/1e6)}MHz_P{pilot_spacing}_equalized.png")
    else:
        fname = f"{M}_{model}_P{pilot_spacing}_equalized.png"

    outpath = os.path.join(out_dir, fname)
    fig.savefig(outpath, dpi = 100)
    plt.close(fig)
    print(f"Saved equalized plot to {outpath}")


