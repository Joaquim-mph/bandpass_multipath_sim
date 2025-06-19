import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Sequence
from utils import QPSK, QAM16

def _build_constellation_lut(M: int):
    """
    Return a list of complex constellation points in index order.
    """
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
    coloring each symbol according to its index (Gray code) with legend.

    Parameters
    ----------
    tx : np.ndarray of complex
        Transmitted symbols.
    sym_indices : Sequence[int]
        Symbol indices [0..M-1] corresponding to tx.
    eqs : list of np.ndarray of complex
        Equalized symbol streams.
    labels : list of str
        Titles for each equalized plot.
    num_pts : int
        Number of points to scatter per subplot.
    """
    M = len(np.unique(sym_indices))
    bps = int(np.log2(M))
    # Prepare colormap with M discrete colors
    cmap = plt.cm.get_cmap('tab20', M)
    norm = mcolors.BoundaryNorm(boundaries=np.arange(M+1)-0.5, ncolors=M)
    # Build ideal constellation LUT for nearest-neighbor mapping
    constellation = _build_constellation_lut(M)

    n_plots = 1 + len(eqs)
    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))

    # Plot TX with symbol colors
    tx_seg = tx
    idx_seg = np.array(sym_indices)
    sc = axes[0].scatter(
        tx_seg.real, tx_seg.imag,
        c=idx_seg, cmap=cmap, norm=norm, s=10
    )
    axes[0].set_title("TX Ideal")
    axes[0].axhline(0); axes[0].axvline(0)
    axes[0].set_xlabel("In‐phase"); axes[0].set_ylabel("Quadrature")
    # Colorbar with binary labels
    cbar = fig.colorbar(sc, ax=axes[0], ticks=np.arange(M))
    cbar.ax.set_yticklabels([f"{i:0{bps}b}" for i in range(M)])

    # Plot each equalized stream
    for ax, eq, label in zip(axes[1:], eqs, labels):
        eq_seg = eq
        # Assign each point to nearest constellation index
        eq_idx = [
            int(np.argmin(np.abs(pt - constellation)))
            for pt in eq_seg
        ]
        ax.scatter(
            eq_seg.real, eq_seg.imag,
            c=eq_idx, cmap=cmap, norm=norm, s=10
        )
        ax.set_title(label)
        ax.axhline(0); ax.axvline(0)
        ax.set_xlabel("In‐phase"); ax.set_ylabel("Quadrature")

    plt.tight_layout()
    plt.show()
