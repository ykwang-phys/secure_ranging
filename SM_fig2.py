#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 18:14:06 2025

@author: yunkaiwang
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import qr
from scipy.special import hermite
import math
from scipy.linalg import eig,eigh,inv
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
    Uprime = np.empty((2,2), dtype=complex)
    Uprime[0,0] = V[0,0]*P[0,0]
    Uprime[1,0] = V[0,1]*Qm[1,0]
    Uprime[0,1] = W[0,1]*P[1,0]
    Uprime[1,1] = W[0,0]*Qm[0,0]

    # Precompute the scalar factors a,b for each (p,q) pair in the products
    # Term with δ11 uses (p,q)=(0,0): a = U*_{0,0} U'_{0,0}, b = U*_{1,0} U'_{1,0}, etc.
    # Eq. B13
    a11, b11 = np.conj(U[0,0]) * Uprime[0,0], np.conj(U[1,0]) * Uprime[1,0]
    a12, b12 = np.conj(U[0,0]) * Uprime[0,1], np.conj(U[1,0]) * Uprime[1,1]
    a21, b21 = np.conj(U[0,1]) * Uprime[0,0], np.conj(U[1,1]) * Uprime[1,0]
    a22, b22 = np.conj(U[0,1]) * Uprime[0,1], np.conj(U[1,1]) * Uprime[1,1]

    # Δy and Ny''
    delta_y = y_prime - y_dprime
    Nypp = N * y_dprime

    # Accumulate the four binomial sums
    T11 = 0+0j
    T12 = 0+0j
    T21 = 0+0j
    T22 = 0+0j

    for Q in range(N+1):
        binom = comb(N, Q)

        # δpq(Q) per Eq. B15 (with Δy = y' - y''):
        d11 = np.exp(-(beta**2) * (Q**2) * (delta_y**2))
        d12 = np.exp(-(beta**2) * ((delta_y*(N - Q) + Nypp)**2))
        d21 = np.exp(-(beta**2) * ((delta_y*Q + Nypp)**2))
        d22 = np.exp(-(beta**2) * ((N - Q)**2) * (delta_y**2))

        # Binomial expansion of ∑_{i⃗} ∏_j (...) = ∑_{Q} C(N,Q) a^{N-Q} b^Q
        T11 += binom * (a11**(N-Q)) * (b11**Q) * d11
        T12 += binom * (a12**(N-Q)) * (b12**Q) * d12
        T21 += binom * (a21**(N-Q)) * (b21**Q) * d21
        T22 += binom * (a22**(N-Q)) * (b22**Q) * d22
    #print ('T',T11,T12,T21,T22)
    # Combine coefficients from Eq. B13
    result = ( (abs(psil)**2) * T11
             + (np.conj(psil)*psir) * T12
             + (psil*np.conj(psir)) * T21
             + (abs(psir)**2) * T22 )

    return result


def P_error_single(P_array,psil_array,psir_array,
                         U, V, W, P, Qm,  # 2x2 complex arrays
                         beta,             # float
                         y_prime, y_dprime,# floats (y' and y'')
                         N):               # integer photon number


    result=0
    
    (l,)=np.shape(P_array)
    
    for i in range(l):
        psil=psil_array[i]
        psir=psir_array[i]
        temp=inner_product_eq_B13(U, V, W, P, Qm, psil, psir, beta, y_prime, y_dprime, N)  
        result+=P_array[i]*np.abs(temp)**2

    
    return result


def u2_from_params(alpha, theta, phi, psi):
    """
    U(2) parameterization: U = e^{i alpha} * SU(2)(theta, phi, psi)
    SU(2) = [[ e^{i phi} cosθ,  e^{i psi} sinθ],
             [ -e^{-i psi} sinθ, e^{-i phi} cosθ]]
    """
    c = np.cos(theta)
    s = np.sin(theta)
    su2 = np.array([[np.exp(1j*phi)*c,  np.exp(1j*psi)*s],
                    [-np.exp(-1j*psi)*s, np.exp(-1j*phi)*c]], dtype=complex)
    return np.exp(1j*alpha) * su2

def _build_mats_from_x(x):
    """
    x packs 16 parameters: [aV,tV,phV,psV, aW,tW,phW,psW, aP,tP,phP,psP, aQ,tQ,phQ,psQ]
    returns V, W, P, Qm
    """
    aV,tV,phV,psV, aW,tW,phW,psW, aP,tP,phP,psP, aQ,tQ,phQ,psQ = x
    V  = u2_from_params(aV, tV, phV, psV)
    W  = u2_from_params(aW, tW, phW, psW)
    Pm = u2_from_params(aP, tP, phP, psP)
    Qm = u2_from_params(aQ, tQ, phQ, psQ)
    return V, W, Pm, Qm

def _objective_x(x, P_array, psil_array, psir_array, U, beta, y_prime, y_dprime, N):
    # Bound-protect (optimizer respects bounds, but be robust)
    # Wrap angles into valid ranges without breaking differentiability too much:
    x = np.array(x, dtype=float)
    # α, φ, ψ in [0, 2π), θ in [0, π/2]
    x[0::4]  = np.mod(x[0::4], 2*np.pi)          # α's
    x[2::4]  = np.mod(x[2::4], 2*np.pi)          # φ's
    x[3::4]  = np.mod(x[3::4], 2*np.pi)          # ψ's
    x[1::4]  = np.clip(x[1::4], 0.0, 0.5*np.pi)  # θ's

    V, W, Pm, Qm = _build_mats_from_x(x)
    val = P_error_single(P_array, psil_array, psir_array, U, V, W, Pm, Qm, beta, y_prime, y_dprime, N)
    # We maximize P_error_single -> minimize negative
    return -float(np.real(val))

def optimize_unitaries_for_max_P_error_single(
        P_array, psil_array, psir_array,
        U, beta, y_prime, y_dprime, N,
        num_restarts=12, random_state=None, ftol=1e-9, gtol=None, maxiter=2000, verbose=False):

    rng = np.random.default_rng(random_state)

    bounds = []
    for _ in range(4):
        bounds += [(0, 2*np.pi), (0, 0.5*np.pi), (0, 2*np.pi), (0, 2*np.pi)]
    bounds = tuple(bounds)

    best = {"value": -np.inf, "V": None, "W": None, "P": None, "Qm": None, "x": None, "result": None}

    # Build optimizer options safely
    opt_kwargs = {
        "method": "L-BFGS-B",
        "bounds": bounds,
    }
    opt_options = {"ftol": ftol, "maxiter": maxiter, "iprint": 1 if verbose else -1}
    if gtol is not None:
        opt_options["gtol"] = float(gtol)  # must be a float
    opt_kwargs["options"] = opt_options

    for r in range(num_restarts):
        x0 = np.empty(16, dtype=float)
        x0[0::4] = rng.uniform(0, 2*np.pi, 4)
        x0[1::4] = rng.uniform(0, 0.5*np.pi, 4)
        x0[2::4] = rng.uniform(0, 2*np.pi, 4)
        x0[3::4] = rng.uniform(0, 2*np.pi, 4)

        res = minimize(
            _objective_x, x0,
            args=(P_array, psil_array, psir_array, U, beta, y_prime, y_dprime, N),
            **opt_kwargs
        )

        Vopt, Wopt, Popt, Qopt = _build_mats_from_x(res.x)
        achieved = float(np.real(P_error_single(
            P_array, psil_array, psir_array, U, Vopt, Wopt, Popt, Qopt, beta, y_prime, y_dprime, N
        )))

        if verbose:
            print(f"[restart {r+1}/{num_restarts}] achieved = {achieved:.12g}, success={res.success}")
            #None
        if achieved > best["value"]:
            best.update({
                "value": achieved,
                "V": Vopt, "W": Wopt, "P": Popt, "Qm": Qopt,
                "x": res.x.copy(),
                "result": res
            })

    return best
'''

#U=np.array([[1,1],[-1,1]])/np.sqrt(2)
U=np.array([[1,2],[-2,1]])/np.sqrt(5)
#U=np.array([[0,1],[1,0]])
beta=1
y_prime=0
y_dprime=0.001
N=7
#P_array=np.array([1,1,1,1,1,1,1,1])
#P_array=P_array/np.sum(P_array)
#psil_array=np.array([np.sqrt(1/3),np.sqrt(1/4),np.sqrt(1/5),np.sqrt(1/7),np.sqrt(1/8),np.sqrt(1/9),np.sqrt(7/8),np.sqrt(8/9)])
#psir_array=np.array([np.sqrt(2/3),np.sqrt(3/4),np.sqrt(4/5),np.sqrt(6/7),np.sqrt(7/8),np.sqrt(8/9),np.sqrt(1/8),np.sqrt(1/9)])

P_array=np.array([1/2,1/2])
psil_array=np.array([np.exp(1j*np.random.random())*np.sqrt(1/2),np.exp(1j*np.random.random())*np.sqrt(1/2)])
psir_array=np.array([np.sqrt(1/2),np.sqrt(1/2)])


best = optimize_unitaries_for_max_P_error_single(
    P_array, psil_array, psir_array,
    U, beta, y_prime, y_dprime, N,
    num_restarts=16, random_state=42, verbose=False
)

print("Max P_error_single =", best["value"])
V_opt, W_opt, P_opt, Qm_opt = best["V"], best["W"], best["P"], best["Qm"]

Uprime = np.empty((2,2), dtype=complex)
Uprime[0,0] = V_opt[0,0]*P_opt[0,0]
Uprime[1,0] = V_opt[0,1]*Qm_opt[1,0]
Uprime[0,1] = W_opt[0,1]*P_opt[1,0]
Uprime[1,1] = W_opt[0,0]*Qm_opt[0,0]

print (Uprime)

(l,)=np.shape(P_array)
temp=0
for i in range(l):
    temp+=P_array[i]*max(np.abs(psil_array[i])**4,np.abs(psir_array[i])**4)
    
print ('max P theoretical',temp)
'''

'''

#U=np.array([[1,2],[-2,1]])/np.sqrt(5)
U=np.array([[1,1],[-1,1]])/np.sqrt(2)
beta=1
y_prime=0
y_dprime=0
P_array=np.array([1/2,1/2])
#psil_array=np.array([np.exp(1j*np.random.random())*np.sqrt(1/2),np.exp(1j*np.random.random())*np.sqrt(1/2)])
#psir_array=np.array([np.sqrt(1/2),np.sqrt(1/2)])
#psil_array=np.array([np.sqrt(1/2),np.exp(1j*np.pi)*np.sqrt(1/2)])
#psir_array=np.array([np.sqrt(1/2),np.sqrt(1/2)])

#psil_array=np.array([np.sqrt(1/3),np.exp(1j*np.pi)*np.sqrt(2/3)])
#psir_array=np.array([np.sqrt(2/3),np.sqrt(1/3)])

psil_array=np.array([np.sqrt(1/4),np.exp(1j*np.pi)*np.sqrt(3/4)])
psir_array=np.array([np.sqrt(3/4),np.sqrt(1/4)])


M=10
N_array=np.linspace(1,M,M)
P_error=np.zeros(M)

for i in range(M):
    N=int(N_array[i])
    best = optimize_unitaries_for_max_P_error_single(
        P_array, psil_array, psir_array,
        U, beta, y_prime, y_dprime, N,
        num_restarts=16, random_state=42, verbose=False
    )
    P_error[i]=best["value"]
    print (N,P_error[i])
    
plt.figure()
plt.plot(N_array,P_error)
plt.show()


(l,)=np.shape(P_array)
temp=0
for i in range(l):
    temp+=P_array[i]*max(np.abs(psil_array[i])**4,np.abs(psir_array[i])**4)
    
print ('max P theoretical',temp)

'''

#U=np.array([[1,3],[-3,1]])/np.sqrt(10)
#U=np.array([[1,2],[-2,1]])/np.sqrt(5)
U=np.array([[1,1j],[1j,1]])/np.sqrt(2)
beta=1
y_prime=0
y_dprime=0
P_array=np.array([1/2,1/2])

M=10
N_array=np.linspace(1,M,M)

psil_array=np.array([np.sqrt(1/2),np.exp(1j*np.pi)*np.sqrt(1/2)])
psir_array=np.array([np.sqrt(1/2),np.sqrt(1/2)])

P_error1=np.zeros(M)

for i in range(M):
    N=int(N_array[i])
    best = optimize_unitaries_for_max_P_error_single(
        P_array, psil_array, psir_array,
        U, beta, y_prime, y_dprime, N,
        num_restarts=16, random_state=42, verbose=False
    )
    P_error1[i]=best["value"]
    print (N,P_error1[i])
    

(l,)=np.shape(P_array)
temp=0
for i in range(l):
    temp+=P_array[i]*max(np.abs(psil_array[i])**4,np.abs(psir_array[i])**4)
    
print ('max P theoretical',temp)



psil_array=np.array([np.sqrt(1/3),np.exp(1j*np.pi)*np.sqrt(2/3)])
psir_array=np.array([np.sqrt(2/3),np.sqrt(1/3)])

P_error2=np.zeros(M)

for i in range(M):
    N=int(N_array[i])
    best = optimize_unitaries_for_max_P_error_single(
        P_array, psil_array, psir_array,
        U, beta, y_prime, y_dprime, N,
        num_restarts=16, random_state=42, verbose=False
    )
    P_error2[i]=best["value"]
    print (N,P_error2[i])
    

(l,)=np.shape(P_array)
temp=0
for i in range(l):
    temp+=P_array[i]*max(np.abs(psil_array[i])**4,np.abs(psir_array[i])**4)
    
print ('max P theoretical',temp)



psil_array=np.array([np.sqrt(1/4),np.exp(1j*np.pi)*np.sqrt(3/4)])
psir_array=np.array([np.sqrt(3/4),np.sqrt(1/4)])

P_error3=np.zeros(M)

for i in range(M):
    N=int(N_array[i])
    best = optimize_unitaries_for_max_P_error_single(
        P_array, psil_array, psir_array,
        U, beta, y_prime, y_dprime, N,
        num_restarts=16, random_state=42, verbose=False
    )
    P_error3[i]=best["value"]
    print (N,P_error3[i])
    


(l,)=np.shape(P_array)
temp=0
for i in range(l):
    temp+=P_array[i]*max(np.abs(psil_array[i])**4,np.abs(psir_array[i])**4)
    
print ('max P theoretical',temp)



plt.figure()
ax=plt.subplot(1,1,1)


#xmajorLocator   = MultipleLocator(2) #å°†xä¸»åˆ»åº¦æ ‡ç­¾è®¾ç½®ä¸º20çš„å€æ•°
#ymajorLocator   = MultipleLocator(1000000) #å°†yä¸»åˆ»åº¦æ ‡ç­¾è®¾ç½®ä¸º20çš„å€æ•°
#ymajorLocator = LogLocator(base=10)  # LogLocator for 10^4n
#xmajorFormatter = FormatStrFormatter('%1.1f') #è®¾ç½®xè½´æ ‡ç­¾æ–‡æœ¬çš„æ ¼å¼
#xminorLocator   = MultipleLocator(1) #å°†xè½´æ¬¡åˆ»åº¦æ ‡ç­¾è®¾ç½®ä¸º5çš„å€æ•°
#ymajorLocator   = MultipleLocator(0.5)
#ax.xaxis.set_major_locator(xmajorLocator)
#ax.xaxis.set_major_formatter(xmajorFormatter)
#ax.yaxis.set_major_locator(ymajorLocator)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
#plt.xlim([5,D_array[-1]+10])
#plt.ylim([0,0.55])
#plt.xlim([0,D_array[-1]+1])
#plt.ylim([1e-6,1e-3])

plt.plot(N_array,P_error1,linewidth=2,linestyle='-',label='11111111')
plt.plot(N_array,P_error2,linewidth=2,linestyle='--',label='22222222')
plt.plot(N_array,P_error3,linewidth=2,linestyle='-.',label='33333333')
plt.legend(fontsize=17,loc='center left', bbox_to_anchor=(1, 0.5))  



foo_fig = plt.gcf() # 'get current figure'
#foo_fig.savefig('P1_U3.jpg', format='jpg', dpi=1000,bbox_inches = 'tight')
    
plt.show()
