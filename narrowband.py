import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numba import njit, prange
from styles import set_plot_style
import scienceplots



# —————————————————————————————————————————————
# Parámetros físicos y de simulación
# —————————————————————————————————————————————
fc     = 2.4e9
lambda_= 3e8/fc
v      = 250/3.6
fmax   = v/lambda_
M      = int(1e5)
t      = np.linspace(0, 0.25, M)

N      = 100000
an     = np.ones(N) * np.sqrt(1/N)
thetan = 2*np.pi * np.random.rand(N)
fDn    = fmax * np.cos(2*np.pi * np.random.rand(N))


# —————————————————————————————————————————————
# Cálculo de H(t) = Σ a_n · exp[j(θ_n − 2π fD_n · t)]
# —————————————————————————————————————————————
@njit(parallel=True, fastmath=True)
def gen_rayleigh(t, an, thetan, fDn):
    M = t.size
    H = np.zeros(M, np.complex128)
    for n in prange(an.size):
        # compute phase for this path
        phase = thetan[n] - 2*np.pi * fDn[n] * t
        H += an[n] * np.exp(1j * phase)
    return H

# First call compiles; subsequent calls are blazing fast
H = gen_rayleigh(t, an, thetan, fDn)


# —————————————————————————————————————————————
# Directorio de salida para las figuras
# —————————————————————————————————————————————
fig_dir = './Figures'
os.makedirs(fig_dir, exist_ok=True)


# —————————————————————————————————————————————
# Computations for plots
# —————————————————————————————————————————————

z   = np.linspace(-5, 5, 100)
Dz  = z[1] - z[0]
pdf_im = np.histogram(H.imag, bins=z, density=True)[0]
pdf_re = np.histogram(H.real, bins=z, density=True)[0]
sigma  = np.sqrt(0.5)
pdf_gauss = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-z**2 / (2 * sigma**2))

# compute empirical histogram
counts_re, edges = np.histogram(H.real, bins=z, density=True)
counts_im, _     = np.histogram(H.imag, bins=z, density=True)

# true bin‐centers
centers = 0.5 * (edges[:-1] + edges[1:])

# theoretical Gaussian at those centers
pdf_gauss = (1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-centers**2/(2*sigma**2))

theta = np.linspace(-1.5*np.pi, 1.5*np.pi, 100)
pdf_th = np.histogram(np.angle(H), bins=theta, density=True)[0]


LINEWIDTH = 1.2
FONTSIZE  = 12

# helper to clean up axes
def polish_ax(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
    ax.tick_params(direction='in', width=1.1, length=5)

# —————————————————————————————————————————————
# 1) Magnitud (dB) y fase de H(t)
# —————————————————————————————————————————————
set_plot_style("default")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

ax1.plot(t, 10*np.log10(np.abs(H)**2), lw=LINEWIDTH)
ax1.set_ylabel(r'Power (dB)', fontsize=FONTSIZE)
ax1.set_title(r'Magnitude and Phase of $H(t)$', fontsize=FONTSIZE+2)
polish_ax(ax1)

ax2.plot(t, np.angle(H), lw=LINEWIDTH)
ax2.set_xlabel(r'Time (s)', fontsize=FONTSIZE)
ax2.set_ylabel(r'Phase (rad)', fontsize=FONTSIZE)
polish_ax(ax2)

fig.tight_layout(pad=1.0)
fig.savefig(os.path.join(fig_dir, 'H_magnitude_phase.pdf'), dpi=300)
plt.close(fig)


# —————————————————————————————————————————————
# 2) PDF de la parte real e imaginaria
# —————————————————————————————————————————————
set_plot_style("prism_rain")

fig, ax = plt.subplots(figsize=(7, 7))

# compute bin‐width
bin_width = centers[1] - centers[0]

# plot histograms for real and imag parts
ax.bar(centers, counts_re, width=bin_width, alpha=0.5,
       label=r'$\mathrm{Re}\{H\}$', edgecolor='black', linewidth=0.5)
ax.bar(centers, counts_im, width=bin_width, alpha=0.5,
       label=r'$\mathrm{Im}\{H\}$', edgecolor='black', linewidth=0.5)

# overlay theoretical Gaussian
ax.plot(centers, pdf_gauss, '--', label='Gaussiana teórica', lw=LINEWIDTH)

# labels, title, legend
ax.set_xlabel('Componentes de la respuesta de canal', fontsize=FONTSIZE)
ax.set_ylabel('PDF', fontsize=FONTSIZE)
ax.set_title('PDF de partes real e imag. de $H(t)$', fontsize=FONTSIZE+2)
ax.legend(frameon=False, fontsize=FONTSIZE)

# polish styling
polish_ax(ax)

fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'H_pdf_real_imag.pdf'), dpi=300)
plt.close(fig)

# —————————————————————————————————————————————
# 3) PDF de la fase
# —————————————————————————————————————————————
set_plot_style("default")

fig, ax = plt.subplots(figsize=(7, 7))

ax.plot(theta[:-1], pdf_th, lw=LINEWIDTH)
ax.set_xlabel(r'Angle (rad)', fontsize=FONTSIZE)
ax.set_ylabel('Probability density', fontsize=FONTSIZE)
ax.set_title('PDF of Phase of $H(t)$', fontsize=FONTSIZE+2)
polish_ax(ax)

fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'H_pdf_phase.pdf'), dpi=300)
plt.close(fig)



r = np.abs(H)  # shape = (M,)

# 1b) Choose bin edges for the histogram:
num_bins = 100
r_max = r.max()
bins = np.linspace(0, r_max, num_bins + 1)

# 1c) Empirical PDF of r
hist_r, edges_r = np.histogram(r, bins=bins, density=True)
centers_r = 0.5 * (edges_r[:-1] + edges_r[1:])   # bin centers


# 1e) Theoretical Rayleigh PDF:
#     f_R(r) = (r / σ²) * exp(−r²/(2σ²)),   for r ≥ 0.
pdf_rayleigh = (centers_r / (sigma**2)) * np.exp(-centers_r**2 / (2 * sigma**2))

# 1f) Plot:
plt.figure(figsize=(6,4))
plt.bar(centers_r, hist_r, width=centers_r[1] - centers_r[0],
        alpha=0.5, label="Empirical |H| histogram", edgecolor='k')
plt.plot(centers_r, pdf_rayleigh, 'r--', lw=2, label="Theoretical Rayleigh PDF")
plt.xlabel("Envelope $r = |H(t)|$")
plt.ylabel("Probability density")
plt.title("Empirical vs. Theoretical Rayleigh Envelope")
plt.legend()
plt.tight_layout()
plt.show()