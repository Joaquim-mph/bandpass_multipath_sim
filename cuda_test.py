import numpy as np
from numba import cuda, float64, complex128
import os

# —————————————————————————————————————————————
# 1) Generar los parámetros (mismo setup que siempre)
# —————————————————————————————————————————————
fc      = 2.4e9
lambda_ = 3e8 / fc
v       = 250 / 3.6
fmax    = v / lambda_

M = int(1e5)
t_host = np.linspace(0, 0.25, M, dtype=np.float64)

N = int(1e4)
an_host     = np.ones(N, dtype=np.float64) * np.sqrt(1.0 / N)
thetan_host = (2 * np.pi * np.random.rand(N)).astype(np.float64)
fDn_host    = (fmax * np.cos(2 * np.pi * np.random.rand(N))).astype(np.float64)

# —————————————————————————————————————————————
# 2) Transferir datos a memoria GPU
# —————————————————————————————————————————————
t_dev      = cuda.to_device(t_host)       # vector de tiempos (M,)
an_dev     = cuda.to_device(an_host)      # amplitudes (N,)
thetan_dev = cuda.to_device(thetan_host)  # fases (N,)
fDn_dev    = cuda.to_device(fDn_host)     # Doppler (N,)

# Reservar espacio para H(t) en GPU
H_dev = cuda.device_array(shape=(M,), dtype=np.complex128)

# —————————————————————————————————————————————
# 3) Definir el kernel CUDA
# —————————————————————————————————————————————
@cuda.jit
def rayleigh_kernel(t, an, thetan, fDn, H_out, N):
    """
    Cada hilo calcula H_out[m] = sum_{n=0..N-1} a_n * exp(j*(theta_n - 2*pi*fDn[n]*t[m])).
    t, an, thetan, fDn, y H_out ya están en memoria de dispositivo.
    """
    m = cuda.grid(1)  # índice global del hilo en dimensión 1
    if m >= H_out.size:
        return

    tm = t[m]                 # t[m]
    const2pi = 2.0 * np.pi

    real_acc = 0.0            # acumulador para parte real
    imag_acc = 0.0            # acumulador para parte imaginaria

    # Sumar sobre todos los N paths
    for n in range(N):
        phi = thetan[n] - const2pi * fDn[n] * tm
        c = np.cos(phi)
        s = np.sin(phi)
        a = an[n]
        real_acc += a * c
        imag_acc += a * s

    H_out[m] = real_acc + 1j * imag_acc


# —————————————————————————————————————————————
# 4) Elegir dimensiones de bloque/rejilla y lanzar el kernel
# —————————————————————————————————————————————
threads_per_block = 256
blocks_per_grid   = (M + threads_per_block - 1) // threads_per_block

# Llamada al kernel: cada bloque tiene 256 hilos; hay blocks_per_grid bloques.
rayleigh_kernel[blocks_per_grid, threads_per_block](
    t_dev, an_dev, thetan_dev, fDn_dev, H_dev, N
)

# Esperar a que termine el kernel
cuda.synchronize()

# Copiar H de vuelta a host
H_host = H_dev.copy_to_host()

# —————————————————————————————————————————————
# 5) (Opcional) Guardar o usar H_host para graficar / histograma, etc.
# —————————————————————————————————————————————
# Ejemplo: calcular y guardar el histograma del sobre |H|
r = np.abs(H_host)
num_bins = 100
r_max = r.max()
bins = np.linspace(0, r_max, num_bins + 1)
hist_r, edges_r = np.histogram(r, bins=bins, density=True)
centers_r = 0.5 * (edges_r[:-1] + edges_r[1:])
sigma = np.sqrt(0.5)
pdf_rayleigh = (centers_r / (sigma**2)) * np.exp(-centers_r**2 / (2 * sigma**2))

import matplotlib.pyplot as plt
fig_dir = "./Figures_GPU"
os.makedirs(fig_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 7))
bin_width = centers_r[1] - centers_r[0]
ax.bar(
    centers_r,
    hist_r,
    width=bin_width,
    alpha=0.6,
    edgecolor='black',
    linewidth=0.5,
    label=r'Histograma empírico de $|H|$'
)
ax.plot(
    centers_r,
    pdf_rayleigh,
    'r--',
    lw=2,
    label=r'PDF Rayleigh teórica'
)
ax.set_xlabel(r'Sobre $r = |H(t)|$', fontsize=12)
ax.set_ylabel('Densidad de probabilidad', fontsize=12)
ax.set_title('Envelope Rayleigh: empírico vs. teórico (GPU)', fontsize=14)
ax.legend(frameon=False, fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)

fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'envelope_rayleigh_gpu.pdf'), dpi=300)
plt.close(fig)

print("Done! El kernel se ejecutó en GPU y la figura se guardó en", fig_dir)
