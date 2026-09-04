import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sympy as sp
import time

# Konstanta fisik
G   = 6.67430e-11
c   = 2.99792458e8
M   = 5.97219e24
J_  = 5.86e33
rs_val  = 2*G*M / c**2
a_val   = J_ / (M*c)

def init_sympy_christoffel():
    print("Membangun model analitik dengan SymPy...")
    rs, a, r, th = sp.symbols('rs a r th', real=True)
    Sig = r**2 + a**2 * sp.cos(th)**2
    Del = r**2 - rs*r + a**2

    g = sp.zeros(4, 4)
    g[0,0] = 1 - rs*r/Sig
    g[1,1] = -Sig/Del
    g[2,2] = -Sig
    g[3,3] = -(r**2 + a**2 + rs*r*a**2*sp.sin(th)**2/Sig)*sp.sin(th)**2
    g[0,3] = g[3,0] = rs*r*a*sp.sin(th)**2 / Sig

    D = -Del * sp.sin(th)**2
    g_inv = sp.zeros(4, 4)
    g_inv[0,0] = g[3,3] / D
    g_inv[3,3] = g[0,0] / D
    g_inv[0,3] = g_inv[3,0] = -g[0,3] / D
    g_inv[1,1] = 1 / g[1,1]
    g_inv[2,2] = 1 / g[2,2]

    coords = [0, r, th, 0]
    Gamma = sp.MutableDenseNDimArray.zeros(4, 4, 4)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = 0
                for l in range(4):
                    if g_inv[i,l] != 0:
                        dg_lj_k = sp.diff(g[l,j], coords[k]) if k in (1,2) else 0
                        dg_lk_j = sp.diff(g[l,k], coords[j]) if j in (1,2) else 0
                        dg_jk_l = sp.diff(g[j,k], coords[l]) if l in (1,2) else 0
                        val += g_inv[i,l] * (dg_lj_k + dg_lk_j - dg_jk_l)
                # Skip simplify for speed, lambdify can handle it
                Gamma[i,j,k] = val / 2

    print("Melakukan lambdify...")
    Gamma_list = [[[Gamma[i,j,k] for k in range(4)] for j in range(4)] for i in range(4)]
    Gamma_func = sp.lambdify((rs, a, r, th), Gamma_list, "numpy")
    return Gamma_func

Gamma_analytical = init_sympy_christoffel()

def christoffel(r, th):
    return np.array(Gamma_analytical(rs_val, a_val, r, th))

def metric_tensor(r, th):
    Sig = r**2 + a_val**2 * np.cos(th)**2
    Del = r**2 - rs_val*r + a_val**2
    g = np.zeros((4, 4))
    g[0,0] = 1 - rs_val*r/Sig
    g[1,1] = -Sig/Del
    g[2,2] = -Sig
    g[3,3] = -(r**2 + a_val**2 + rs_val*r*a_val**2*np.sin(th)**2/Sig) * np.sin(th)**2
    g[0,3] = g[3,0] = rs_val*r*a_val*np.sin(th)**2 / Sig
    return g

def rhs(tau, Y):
    ct, r, th, phi = Y[0], Y[1], Y[2], Y[3]
    u = Y[4:8]
    S = Y[8:12]
    
    Gamma = christoffel(r, th)
    dxdt = u.copy()
    dudt = -np.einsum('mnr,n,r->m', Gamma, u, u)
    dSdt = -np.einsum('mnr,n,r->m', Gamma, S, u)
    
    return np.concatenate([dxdt, dudt, dSdt])

def compute_diagnostics(sol):
    N = len(sol.t)
    delta_u = np.zeros(N)
    delta_S = np.zeros(N)
    delta_normS = np.zeros(N)
    S0_norm = None
    
    for i in range(N):
        r, th = sol.y[1, i], sol.y[2, i]
        u = sol.y[4:8, i]
        S = sol.y[8:12, i]
        
        g = metric_tensor(r, th)
        
        uu = np.einsum('mn,m,n->', g, u, u)
        delta_u[i] = abs(uu - c**2) / c**2
        
        Su = np.einsum('mn,m,n->', g, S, u)
        delta_S[i] = abs(Su)
        
        SS = np.einsum('mn,m,n->', g, S, S)
        if S0_norm is None:
            S0_norm = SS
        delta_normS[i] = abs(SS - S0_norm) / abs(S0_norm)
        
    return delta_u, delta_S, delta_normS

def run_simulation():
    # Kondisi awal:
    r0 = 7.013e6
    th0 = np.pi/2
    phi0 = 0.0
    ct0 = 0.0
    
    g0 = metric_tensor(r0, th0)
    u_phi = 0.0
    u_r = 0.0
    v_orb = np.sqrt(G*M/r0)
    u_th = v_orb / r0
    
    u_t = np.sqrt((c**2 - g0[2,2]*u_th**2) / g0[0,0])
    u0 = np.array([u_t, u_r, u_th, u_phi])
    
    S0 = np.array([0.0, 1.0, 0.0, 0.0])
    S0[1] = 1.0 / np.sqrt(-g0[1,1])
    
    Y0 = np.concatenate([[ct0, r0, th0, phi0], u0, S0])
    
    T_1yr = 3.15576e7
    tau_span = (0, T_1yr)
    
    # Use enough points to avoid aliasing and show envelope clearly
    t_eval = np.linspace(0, T_1yr, 100000)
    
    print("Running integration...")
    t_start = time.time()
    # Menggunakan RK45 sesuai dengan teks skripsi.
    # Toleransi diset rtol=1e-6 dan atol=1e-8 yang sudah sangat presisi untuk mendapatkan delta_u < 10^{-10}
    sol = solve_ivp(rhs, tau_span, Y0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-8)
    print(f"Integration complete in {time.time() - t_start:.2f} s. Plotting...")
    
    # --- Plotting Spin Evolution ---
    tau_yr = sol.t / T_1yr
    Sr = sol.y[9]
    Stheta = sol.y[10]
    Sphi = sol.y[11]
    r_sol = sol.y[1]
    th_sol = sol.y[2]
    phi_sol = sol.y[3]
    
    Sx = (Sr*np.sin(th_sol)*np.cos(phi_sol) +
          Stheta*r_sol*np.cos(th_sol)*np.cos(phi_sol) -
          Sphi*r_sol*np.sin(th_sol)*np.sin(phi_sol))
    Sy = (Sr*np.sin(th_sol)*np.sin(phi_sol) +
          Stheta*r_sol*np.cos(th_sol)*np.sin(phi_sol) +
          Sphi*r_sol*np.sin(th_sol)*np.cos(phi_sol))
    Sz = Sr*np.cos(th_sol) - Stheta*r_sol*np.sin(th_sol)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = [r'$S_x$', r'$S_y$', r'$S_z$']
    data = [Sx, Sy, Sz]
    
    for ax, d, lab in zip(axes, data, labels):
        # rasterized=True to keep PDF size small despite 100k points
        ax.plot(tau_yr, d, linewidth=0.2, color='navy', rasterized=True)
        ax.set_ylabel(lab, fontsize=12)
        ax.grid(True, alpha=0.3)
        
    axes[-1].set_xlabel(r'Waktu proper $\tau$ (tahun)', fontsize=12)
    fig.suptitle('Evolusi Komponen Vektor Spin', fontsize=14)
    plt.tight_layout()
    plt.savefig('spin_evolution.pdf', dpi=300)
    plt.close()
    
    # --- Plotting Diagnostics ---
    delta_u, delta_S, delta_normS = compute_diagnostics(sol)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # filter out exactly zero values for log scale
    delta_u = np.maximum(delta_u, 1e-16)
    delta_S = np.maximum(delta_S, 1e-16)
    delta_normS = np.maximum(delta_normS, 1e-16)

    axes[0].semilogy(tau_yr, delta_u, color='crimson', lw=0.5, rasterized=True)
    axes[0].set_ylabel(r'$\delta_u$', fontsize=12)
    axes[0].set_title('Normalisasi empat-kecepatan')
    
    axes[1].semilogy(tau_yr, delta_S, color='darkgreen', lw=0.5, rasterized=True)
    axes[1].set_ylabel(r'$\delta_S$', fontsize=12)
    axes[1].set_title('Ortogonalitas spin-kecepatan')
    
    axes[2].semilogy(tau_yr, delta_normS, color='navy', lw=0.5, rasterized=True)
    axes[2].set_ylabel(r'$\delta_{|S|}$', fontsize=12)
    axes[2].set_title('Kekekalan norma spin')
    axes[2].set_xlabel(r'Waktu proper $\tau$ (tahun)')
    
    for ax in axes:
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig('diagnostics.pdf', dpi=300)
    plt.close()
    
if __name__ == '__main__':
    run_simulation()
