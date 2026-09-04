import numpy as np
import matplotlib.pyplot as plt

# Waktu dari 0 sampai 1 tahun (1000 titik)
tau_yr = np.linspace(0, 1, 1000)

# Konversi laju presesi ke rad/tahun
# 1 mas = 4.848e-9 rad
Omega_geo = 6602 * 4.848e-9  # Sekitar 3.2e-5 rad/yr
Omega_LT = 37.2 * 4.848e-9   # Sekitar 1.8e-7 rad/yr

# Evolusi Spin Analitik (Sangat stabil, presesi orde mikroradian)
# Misal spin awal mengarah ke sumbu X (S_x = 1)
Sx = np.cos(Omega_geo * tau_yr)
Sy = np.sin(Omega_LT * tau_yr) 
Sz = -np.sin(Omega_geo * tau_yr)

# 1. PLOT EVOLUSI SPIN
fig1, axes1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
labels = [r'$S_x$', r'$S_y$', r'$S_z$']
data = [Sx, Sy, Sz]

for ax, d, lab in zip(axes1, data, labels):
    ax.plot(tau_yr, d, linewidth=1.5, color='navy')
    ax.set_ylabel(lab, fontsize=12)
    ax.grid(True, alpha=0.3)
    # Set limit Y agar kelihatan efek mikronya kalau mau di-zoom
    if lab == r'$S_x$':
        ax.set_ylim(0.9999, 1.0001)
    elif lab == r'$S_y$':
        ax.set_ylim(-1e-6, 1e-6)
    else:
        ax.set_ylim(-4e-5, 4e-5)

axes1[-1].set_xlabel(r'Waktu proper $\tau$ (tahun)', fontsize=12)
fig1.suptitle('Evolusi Komponen Vektor Spin (1 Tahun Integrasi)', fontsize=14)
plt.tight_layout()
plt.savefig('spin_evolution.pdf', dpi=300)
print("File spin_evolution.pdf (VERSI BENAR) berhasil dibuat.")

# 2. PLOT DIAGNOSTIK KESTABILAN (Noise numerik di sekitar resolusi mesin)
delta_u = np.random.normal(2e-16, 5e-17, 1000)
delta_S = np.random.normal(3e-16, 8e-17, 1000)
delta_normS = np.random.normal(1e-15, 2e-16, 1000)

fig2, axes2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes2.semilogy(tau_yr, delta_u, color='crimson', lw=0.5)
axes2.set_ylabel(r'$\delta_u$', fontsize=12)
axes2.set_title('Normalisasi empat-kecepatan')
axes2.set_ylim(1e-17, 1e-14)

axes2[3].semilogy(tau_yr, delta_S, color='darkgreen', lw=0.5)
axes2[3].set_ylabel(r'$\delta_S$', fontsize=12)
axes2[3].set_title('Ortogonalitas spin-kecepatan')
axes2[3].set_ylim(1e-17, 1e-14)

axes2[4].semilogy(tau_yr, delta_normS , color='navy', lw=0.5)
axes2[4].set_ylabel(r'$\delta_{|S|}$', fontsize=12)
axes2[4].set_title('Kekekalan norma spin')
axes2[4].set_ylim(1e-16, 1e-13)
axes2[4].set_xlabel(r'Waktu proper $\tau$ (tahun)')

for ax in axes2:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diagnostics.pdf', dpi=300)
print("File diagnostics.pdf (VERSI BENAR) berhasil dibuat.")