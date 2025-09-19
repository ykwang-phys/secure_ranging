#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 19:58:00 2025

@author: yunkaiwang
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import qr
from scipy.special import hermite
import math
from scipy.linalg import eig, eigh, inv
import time
from matplotlib.ticker import MultipleLocator
from scipy import integrate
from math import comb
from scipy.optimize import minimize


def inner_product_eq_B13(U, V, W, P, Qm,  # 2x2 complex arrays
                         psil, psir,       # complex amplitudes
                         beta,             # float
                         y_prime, y_dprime,# floats (y' and y'')
                         N):               # integer photon number
    """
    Compute ⟨ϕ_{y''} | γ_{y'}⟩ (Eq. B13) for Gaussian pulses (Eq. B15).

    Parameters
    ----------
    U, V, W, P, Qm : (2,2) complex ndarrays
        2×2 unitaries; Qm is the matrix called 'Q' in the paper.
    psil, psir : complex
        Amplitudes ψ_l and ψ_r.
    beta : float
        Spectral width parameter β.
    y_prime : float
        Fake position y'.
    y_dprime : float
        Estimated position y''.
    N : int
        Photon number.

    Returns
    -------
    complex
        The inner product ⟨ϕ_{y''} | γ_{y'}⟩.
    """

    # Build U' from cheaters' optics: U'00=V00P00, U'10=V01Q10, U'01=W01P10, U'11=W00Q00
    Uprime = np.empty((2, 2), dtype=complex)
    Uprime[0, 0] = V[0, 0] * P[0, 0]
    Uprime[1, 0] = V[0, 1] * Qm[1, 0]
    Uprime[0, 1] = W[0, 1] * P[1, 0]
    Uprime[1, 1] = W[0, 0] * Qm[0, 0]

    # Precompute the scalar factors a,b for each (p,q) pair in the products
    # Term with δ11 uses (p,q)=(0,0): a = U*_{0,0} U'_{0,0}, b = U*_{1,0} U'_{1,0}, etc.
    # Eq. C23
    a11, b11 = np.conj(Uprime[0, 0]) * Uprime[0, 0], np.conj(Uprime[1, 0]) * Uprime[1, 0]
    a12, b12 = np.conj(Uprime[0, 0]) * Uprime[0, 1], np.conj(Uprime[1, 0]) * Uprime[1, 1]
    a21, b21 = np.conj(Uprime[0, 1]) * Uprime[0, 0], np.conj(Uprime[1, 1]) * Uprime[1, 0]
    a22, b22 = np.conj(Uprime[0, 1]) * Uprime[0, 1], np.conj(Uprime[1, 1]) * Uprime[1, 1]

    # Δy and Ny''
    delta_y = y_prime - y_dprime
    Nypp = N * y_dprime

    # Accumulate the four binomial sums
    T11 = 0 + 0j
    T12 = 0 + 0j
    T21 = 0 + 0j
    T22 = 0 + 0j

    for Q in range(N + 1):
        binom = comb(N, Q)

        # δpq(Q) per Eq. B15 (with Δy = y' - y''):
        d11 = np.exp(-(beta**2) * (Q**2) * (delta_y**2))
        d12 = np.exp(-(beta**2) * ((delta_y * (N - Q) + Nypp)**2))
        d21 = np.exp(-(beta**2) * ((delta_y * Q + Nypp)**2))
        d22 = np.exp(-(beta**2) * (((N - Q)**2) * (delta_y**2)))

        # Binomial expansion of ∑_{i⃗} ∏_j (...) = ∑_{Q} C(N,Q) a^{N-Q} b^Q
        T11 += binom * (a11**(N - Q)) * (b11**Q) * d11
        T12 += binom * (a12**(N - Q)) * (b12**Q) * d12
        T21 += binom * (a21**(N - Q)) * (b21**Q) * d21
        T22 += binom * (a22**(N - Q)) * (b22**Q) * d22

    # Combine coefficients from Eq. B13
    result = ((abs(psil)**2) * T11
              + (np.conj(psil) * psir) * T12
              + (psil * np.conj(psir)) * T21
              + (abs(psir)**2) * T22)

    return result


def P_error_single(P_array, psil_array, psir_array,
                   U, V, W, P, Qm,  # 2x2 complex arrays
                   beta,             # float
                   y_prime, y_dprime,# floats (y' and y'')
                   N):               # integer photon number

    result = 0
    (l,) = np.shape(P_array)

    for i in range(l):
        psil = psil_array[i]
        psir = psir_array[i]
        temp = inner_product_eq_B13(U, V, W, P, Qm, psil, psir, beta, y_prime, y_dprime, N)
        result += P_array[i] * np.abs(temp)

    return result


def u2_from_params(alpha, theta, phi, psi):
    """
    U(2) parameterization: U = e^{i alpha} * SU(2)(theta, phi, psi)
    SU(2) = [[ e^{i phi} cosθ,  e^{i psi} sinθ],
             [ -e^{-i psi} sinθ, e^{-i phi} cosθ]]
    """
    c = np.cos(theta)
    s = np.sin(theta)
    su2 = np.array([[np.exp(1j * phi) * c,  np.exp(1j * psi) * s],
                    [-np.exp(-1j * psi) * s, np.exp(-1j * phi) * c]], dtype=complex)
    return np.exp(1j * alpha) * su2


def _build_mats_from_x(x):
    """
    x packs 16 parameters: [aV,tV,phV,psV, aW,tW,phW,psW, aP,tP,phP,psP, aQ,tQ,phQ,psQ]
    returns V, W, P, Qm
    """
    aV, tV, phV, psV, aW, tW, phW, psW, aP, tP, phP, psP, aQ, tQ, phQ, psQ = x
    V = u2_from_params(aV, tV, phV, psV)
    W = u2_from_params(aW, tW, phW, psW)
    Pm = u2_from_params(aP, tP, phP, psP)
    Qm = u2_from_params(aQ, tQ, phQ, psQ)
    return V, W, Pm, Qm


def _uprime_from(V, W, Pm, Qm):
    Uprime = np.empty((2, 2), dtype=complex)
    Uprime[0, 0] = V[0, 0] * Pm[0, 0]
    Uprime[1, 0] = V[0, 1] * Qm[1, 0]
    Uprime[0, 1] = W[0, 1] * Pm[1, 0]
    Uprime[1, 1] = W[0, 0] * Qm[0, 0]
    return Uprime

def _cval_from_uprime(Uprime):
    return float(max(min(abs(Uprime[0, 0]), abs(Uprime[1, 0])),
                     min(abs(Uprime[0, 1]), abs(Uprime[1, 1]))))



def _objective_x(x, P_array, psil_array, psir_array, U, beta, y_prime, y_dprime, N, epsilon,
                 penalty_scale=1e6):
    # Wrap angles into valid ranges:
    x = np.array(x, dtype=float)
    x[0::4] = np.mod(x[0::4], 2 * np.pi)          # α
    x[2::4] = np.mod(x[2::4], 2 * np.pi)          # φ
    x[3::4] = np.mod(x[3::4], 2 * np.pi)          # ψ
    x[1::4] = np.clip(x[1::4], 0.0, 0.5 * np.pi)  # θ

    V, W, Pm, Qm = _build_mats_from_x(x)

    # Nonlinear constraint via penalty
    Uprime = _uprime_from(V, W, Pm, Qm)
    cval = _cval_from_uprime(Uprime)
    gap = max(0.0, float(epsilon - cval))
    penalty = penalty_scale * (gap * gap)

    val = P_error_single(P_array, psil_array, psir_array, U, V, W, Pm, Qm,
                         beta, y_prime, y_dprime, N)
    # maximize val -> minimize -val; add penalty
    return -float(np.real(val)) + penalty


def optimize_unitaries_for_max_P_error_single(
        P_array, psil_array, psir_array,
        U, beta, y_prime, y_dprime, N, epsilon,
        num_restarts=12, random_state=None, ftol=1e-9, gtol=None, maxiter=2000, verbose=False):

    rng = np.random.default_rng(random_state)

    bounds = []
    for _ in range(4):
        bounds += [(0, 2 * np.pi), (0, 0.5 * np.pi), (0, 2 * np.pi), (0, 2 * np.pi)]
    bounds = tuple(bounds)

    # Track best feasible and best infeasible (fallback) solutions
    best_feas = {"value": -np.inf, "V": None, "W": None, "P": None, "Qm": None, "x": None,
                 "result": None, "feasible": True, "cval": -np.inf}
    best_any  = {"value": -np.inf, "V": None, "W": None, "P": None, "Qm": None, "x": None,
                 "result": None, "feasible": False, "cval": -np.inf}

    # Optimizer options
    opt_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
    opt_options = {"ftol": ftol, "maxiter": maxiter, "iprint": 1 if verbose else -1}
    if gtol is not None:
        opt_options["gtol"] = float(gtol)
    opt_kwargs["options"] = opt_options

    # To avoid degenerate starts (very small sin/cos), avoid θ near 0 or π/2
    theta_low  = 0.02
    theta_high = 0.5 * np.pi - 0.02

    for r in range(num_restarts):
        x0 = np.empty(16, dtype=float)
        x0[0::4] = rng.uniform(0, 2 * np.pi, 4)                 # α
        x0[1::4] = rng.uniform(theta_low, theta_high, 4)        # θ
        x0[2::4] = rng.uniform(0, 2 * np.pi, 4)                 # φ
        x0[3::4] = rng.uniform(0, 2 * np.pi, 4)                 # ψ

        res = minimize(
            _objective_x, x0,
            args=(P_array, psil_array, psir_array, U, beta, y_prime, y_dprime, N, epsilon),
            **opt_kwargs
        )

        Vopt, Wopt, Popt, Qopt = _build_mats_from_x(res.x)
        Uprime = _uprime_from(Vopt, Wopt, Popt, Qopt)
        cval = _cval_from_uprime(Uprime)

        achieved = float(np.real(P_error_single(
            P_array, psil_array, psir_array, U, Vopt, Wopt, Popt, Qopt, beta, y_prime, y_dprime, N
        )))

        is_feasible = (cval > epsilon)

        if verbose:
            print(f"[restart {r+1}/{num_restarts}] achieved={achieved:.12g}, cval={cval:.6g}, "
                  f"feasible={is_feasible}, success={res.success}")

        # Update best-any (fallback): prefer larger cval; break ties by achieved value
        if (cval > best_any["cval"]) or (np.isclose(cval, best_any["cval"]) and achieved > best_any["value"]):
            best_any.update({
                "value": achieved, "V": Vopt, "W": Wopt, "P": Popt, "Qm": Qopt,
                "x": res.x.copy(), "result": res, "feasible": False, "cval": cval
            })

        # Update best-feasible by achieved value
        if is_feasible and achieved > best_feas["value"]:
            best_feas.update({
                "value": achieved, "V": Vopt, "W": Wopt, "P": Popt, "Qm": Qopt,
                "x": res.x.copy(), "result": res, "feasible": True, "cval": cval
            })

    # Return feasible if we found one; else fallback (non-None matrices)
    return best_feas if best_feas["V"] is not None else best_any


'''
# -----------------------------
# Example usage / experiment 1
# -----------------------------

U = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
#U = np.array([[1, 2], [-2, 1]]) / np.sqrt(5)
#U = np.array([[0, 1], [1, 0]])
beta = 1
y_prime = 0
y_dprime = 0.0
N =2

P_array = np.array([1/2, 1/2])
psil_array = np.array([np.exp(1j * np.random.random()) * np.sqrt(1/2),
                       np.exp(1j * np.random.random()) * np.sqrt(1/2)])
psir_array = np.array([np.sqrt(1/2), np.sqrt(1/2)])

# NEW: epsilon constraint parameter
epsilon = 0.10  # set as desired

best = optimize_unitaries_for_max_P_error_single(
    P_array, psil_array, psir_array,
    U, beta, y_prime, y_dprime, N, epsilon,
    num_restarts=16, random_state=42, verbose=False
)

print("Max P_error_single =", best["value"])
V_opt, W_opt, P_opt, Qm_opt = best["V"], best["W"], best["P"], best["Qm"]

Uprime = np.empty((2, 2), dtype=complex)
Uprime[0, 0] = V_opt[0, 0] * P_opt[0, 0]
Uprime[1, 0] = V_opt[0, 1] * Qm_opt[1, 0]
Uprime[0, 1] = W_opt[0, 1] * P_opt[1, 0]
Uprime[1, 1] = W_opt[0, 0] * Qm_opt[0, 0]

print("Uprime =\n", Uprime)


'''
'''
# -----------------------------
# Additional experiments (optional)
# -----------------------------

# Experiment 2: sweep over N
U = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
beta = 1
y_prime = 0
y_dprime = 0

epsilon=0.05

P_array = np.array([1/2, 1/2])


#psil_array = np.array([np.sqrt(1/4), np.exp(1j*np.pi)*np.sqrt(3/4)])
#psir_array = np.array([np.sqrt(3/4), np.sqrt(1/4)])
psil_array = np.array([ np.sqrt(1/2),np.exp(1j * np.pi/2) * np.sqrt(1/2)])
psir_array = np.array([np.sqrt(1/2), np.sqrt(1/2)])


M = 10
N_array = np.arange(1, M+1)
P_error = np.zeros(M)

for i in range(M):
    N = int(N_array[i])
    best = optimize_unitaries_for_max_P_error_single(
        P_array, psil_array, psir_array,
        U, beta, y_prime, y_dprime, N, epsilon,
        num_restarts=16, random_state=42, verbose=False
    )
    P_error[i] = best["value"]
    print(N, P_error[i])

plt.figure()
plt.plot(N_array, P_error)
plt.show()
'''


# Experiment 3: compare three psi choices
U = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
beta = 1
y_prime = 0
y_dprime = 0
P_array = np.array([1])

M = 10
N_array = np.arange(1, M+1)

psil_array = np.array([np.sqrt(1/4)])
psir_array = np.array([np.sqrt(3/4)])



#psil_array = np.array([np.sqrt(1/4), np.exp(1j*np.pi/2)*np.sqrt(3/4)])
#psir_array = np.array([np.sqrt(3/4), np.sqrt(1/4)])

epsilon=0.02


P_error1 = np.zeros(M)
for i in range(M):
    N = int(N_array[i])
    best = optimize_unitaries_for_max_P_error_single(
        P_array, psil_array, psir_array,
        U, beta, y_prime, y_dprime, N, epsilon,
        num_restarts=16, random_state=42, verbose=False
    )
    P_error1[i] = best["value"]
    print(N, P_error1[i])
    V_opt, W_opt, P_opt, Qm_opt = best["V"], best["W"], best["P"], best["Qm"]

    Uprime = np.empty((2, 2), dtype=complex)
    Uprime[0, 0] = V_opt[0, 0] * P_opt[0, 0]
    Uprime[1, 0] = V_opt[0, 1] * Qm_opt[1, 0]
    Uprime[0, 1] = W_opt[0, 1] * P_opt[1, 0]
    Uprime[1, 1] = W_opt[0, 0] * Qm_opt[0, 0]

    print("Uprime =\n", Uprime)

(l,) = np.shape(P_array)
temp = 0
for i in range(l):
    temp += P_array[i]*max(np.abs(psil_array[i])**4, np.abs(psir_array[i])**4)
print('max P theoretical', temp)




epsilon=0.05


P_error2 = np.zeros(M)
for i in range(M):
    N = int(N_array[i])
    best = optimize_unitaries_for_max_P_error_single(
        P_array, psil_array, psir_array,
        U, beta, y_prime, y_dprime, N, epsilon,
        num_restarts=16, random_state=42, verbose=False
    )
    P_error2[i] = best["value"]
    print(N, P_error2[i])
    V_opt, W_opt, P_opt, Qm_opt = best["V"], best["W"], best["P"], best["Qm"]

    Uprime = np.empty((2, 2), dtype=complex)
    Uprime[0, 0] = V_opt[0, 0] * P_opt[0, 0]
    Uprime[1, 0] = V_opt[0, 1] * Qm_opt[1, 0]
    Uprime[0, 1] = W_opt[0, 1] * P_opt[1, 0]
    Uprime[1, 1] = W_opt[0, 0] * Qm_opt[0, 0]

    print("Uprime =\n", Uprime)

(l,) = np.shape(P_array)
temp = 0
for i in range(l):
    temp += P_array[i]*max(np.abs(psil_array[i])**4, np.abs(psir_array[i])**4)
print('max P theoretical', temp)


epsilon=0.1

P_error3 = np.zeros(M)
for i in range(M):
    N = int(N_array[i])
    best = optimize_unitaries_for_max_P_error_single(
        P_array, psil_array, psir_array,
        U, beta, y_prime, y_dprime, N, epsilon,
        num_restarts=16, random_state=42, verbose=False
    )
    P_error3[i] = best["value"]
    print(N, P_error3[i])
    V_opt, W_opt, P_opt, Qm_opt = best["V"], best["W"], best["P"], best["Qm"]

    Uprime = np.empty((2, 2), dtype=complex)
    Uprime[0, 0] = V_opt[0, 0] * P_opt[0, 0]
    Uprime[1, 0] = V_opt[0, 1] * Qm_opt[1, 0]
    Uprime[0, 1] = W_opt[0, 1] * P_opt[1, 0]
    Uprime[1, 1] = W_opt[0, 0] * Qm_opt[0, 0]

    print("Uprime =\n", Uprime)

(l,) = np.shape(P_array)
temp = 0
for i in range(l):
    temp += P_array[i]*max(np.abs(psil_array[i])**4, np.abs(psir_array[i])**4)
print('max P theoretical', temp)

plt.figure()
ax = plt.subplot(1, 1, 1)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

plt.plot(N_array, P_error1, linewidth=2, linestyle='-', label='psi set 1')
plt.plot(N_array, P_error2, linewidth=2, linestyle='--', label='psi set 2')
plt.plot(N_array, P_error3, linewidth=2, linestyle='-.', label='psi set 3')
plt.legend(fontsize=17, loc='center left', bbox_to_anchor=(1, 0.5))

foo_fig = plt.gcf()
#foo_fig.savefig('P_epsilon.jpg', format='jpg', dpi=1000, bbox_inches='tight')
plt.show()
