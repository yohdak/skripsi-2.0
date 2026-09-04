import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
import time
import os

# --- Parameter Fisika ---
G = 6.67430e-11
c = 2.99792458e8
M = 5.97219e24
J_ = 5.86e33
rs_val = 2 * G * M / c**2
a_val = J_ / (M * c)

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

def run_1year():
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
    t_eval = np.linspace(0, T_1yr, 100000)
    
    print("Running integration 1 year (RK45)...")
    t_start = time.time()
    sol = solve_ivp(rhs, tau_span, Y0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-8)
    print(f"Integration complete in {time.time() - t_start:.2f} s.")
    
    np.savez('data_1year.npz', t=sol.t, y=sol.y)
    print("Data saved to data_1year.npz")

if __name__ == '__main__':
    run_1year()
