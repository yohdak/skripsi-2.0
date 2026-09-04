import numpy as np
import matplotlib.pyplot as plt
import os

# Membaca data mentah 1 tahun
if not os.path.exists('data_1year.npz'):
    print("Menunggu data_1year.npz selesai di-generate...")
    exit()

data = np.load('data_1year.npz')
tau = data['t']
y = data['y']

tau_yr = tau / 3.15576e7
Sr, Stheta, Sphi = y[9], y[10], y[11]
r_sol, th_sol, phi_sol = y[1], y[2], y[3]

Sx = (Sr*np.sin(th_sol)*np.cos(phi_sol) +
      Stheta*r_sol*np.cos(th_sol)*np.cos(phi_sol) -
      Sphi*r_sol*np.sin(th_sol)*np.sin(phi_sol))
Sy = (Sr*np.sin(th_sol)*np.sin(phi_sol) +
      Stheta*r_sol*np.cos(th_sol)*np.sin(phi_sol) +
      Sphi*r_sol*np.sin(th_sol)*np.cos(phi_sol))
Sz = Sr*np.cos(th_sol) - Stheta*r_sol*np.sin(th_sol)

# Prediksi Analitik 1 Tahun
Omega_geo = 6602 * (np.pi / (180 * 3600)) / 1000 # rad/yr
Omega_LT = 37.2 * (np.pi / (180 * 3600)) / 1000  # rad/yr

Sx_ana = np.cos(Omega_geo * tau_yr)
Sy_ana = np.sin(Omega_LT * tau_yr)
Sz_ana = -np.sin(Omega_geo * tau_yr)

# Plot Spin 1 Year Comparison
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes[0].plot(tau_yr, Sx, label='Numerik (RK45)', color='crimson', lw=1, alpha=0.8, rasterized=True)
axes[0].plot(tau_yr, Sx_ana, '--', label='Analitik (Prediksi Teoritis)', color='k', lw=2)
axes[0].set_ylabel(r'$S_x$', fontsize=12)
axes[0].legend(loc='upper right')

axes[1].plot(tau_yr, Sy, color='crimson', lw=1, alpha=0.8, rasterized=True)
axes[1].plot(tau_yr, Sy_ana, '--', color='k', lw=2)
axes[1].set_ylabel(r'$S_y$', fontsize=12)

axes[2].plot(tau_yr, Sz, color='crimson', lw=1, alpha=0.8, rasterized=True)
axes[2].plot(tau_yr, Sz_ana, '--', color='k', lw=2)
axes[2].set_ylabel(r'$S_z$', fontsize=12)
axes[2].set_xlabel(r'Waktu proper $\tau$ (tahun)', fontsize=12)

fig.suptitle('Kegagalan RK45: Numerik vs Analitik (1 Tahun)', fontsize=14)
for ax in axes: ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('spin_evolution_1year_compare.pdf', dpi=300)

print("File spin_evolution_1year_compare.pdf berhasil dibuat.")

# --- Plotting Diagnostics 1 Year ---
def metric_tensor(r, th):
    G = 6.67430e-11
    c = 2.99792458e8
    M = 5.97219e24
    J_ = 5.86e33
    rs = 2 * G * M / c**2
    a = J_ / (M * c)
    Sig = r**2 + a**2 * np.cos(th)**2
    Del = r**2 - rs*r + a**2
    g = np.zeros((4, 4))
    g[0,0] = 1 - rs*r/Sig
    g[1,1] = -Sig/Del
    g[2,2] = -Sig
    g[3,3] = -(r**2 + a**2 + rs*r*a**2*np.sin(th)**2/Sig) * np.sin(th)**2
    g[0,3] = g[3,0] = rs*r*a*np.sin(th)**2 / Sig
    return g

N = len(tau)
delta_u = np.zeros(N)
delta_S = np.zeros(N)
delta_normS = np.zeros(N)
S0_norm = None
c = 2.99792458e8

for i in range(N):
    r_i, th_i = y[1, i], y[2, i]
    u_i = y[4:8, i]
    S_i = y[8:12, i]
    
    g = metric_tensor(r_i, th_i)
    
    uu = np.einsum('mn,m,n->', g, u_i, u_i)
    delta_u[i] = abs(uu - c**2) / c**2
    
    Su = np.einsum('mn,m,n->', g, S_i, u_i)
    delta_S[i] = abs(Su)
    
    SS = np.einsum('mn,m,n->', g, S_i, S_i)
    if S0_norm is None:
        S0_norm = SS
    delta_normS[i] = abs(SS - S0_norm) / abs(S0_norm)

fig2, axes2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

delta_u = np.maximum(delta_u, 1e-16)
delta_S = np.maximum(delta_S, 1e-16)
delta_normS = np.maximum(delta_normS, 1e-16)

axes2[0].semilogy(tau_yr, delta_u, color='crimson', lw=1, rasterized=True)
axes2[0].set_ylabel(r'$\delta_u$', fontsize=12)
axes2[0].set_title('Kegagalan Normalisasi empat-kecepatan (1 Tahun)')

axes2[1].semilogy(tau_yr, delta_S, color='darkgreen', lw=1, rasterized=True)
axes2[1].set_ylabel(r'$\delta_S$', fontsize=12)
axes2[1].set_title('Kegagalan Ortogonalitas spin-kecepatan (1 Tahun)')

axes2[2].semilogy(tau_yr, delta_normS, color='navy', lw=1, rasterized=True)
axes2[2].set_ylabel(r'$\delta_{|S|}$', fontsize=12)
axes2[2].set_title('Pelanggaran Kekekalan norma spin (1 Tahun)')
axes2[2].set_xlabel(r'Waktu proper $\tau$ (tahun)')

for ax in axes2: ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('diagnostics_1year.pdf', dpi=300)
print("File diagnostics_1year.pdf berhasil dibuat.")
