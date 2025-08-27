#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 09:22:17 2025

Author: Rongxing Xu @uestc
Purpose: Modeling defect lines in LC

"""

import numpy as np
from ults_LC import Q2n, n2Q, dirichlet_boundary_conditions, pseudo_grad_energy_elastic, pseudo_grad_energy_LdG



def initial_director_field(X, Y, Z):
    nx0 = 0*np.ones_like(X)
    ny0 = 1*np.ones_like(X)
    nz0 = 0*np.ones_like(X)

    return np.array([nx0, ny0, nz0])



Lx, Ly, Lz = 4, 4, 1
Nx, Ny, Nz = 129, 129, 33
h = Lx / (Nx - 1)
xc = Lx/2
yc = Ly/2
zc = Lz/2
dv = h*h*h
q_tpc = 1
phi_i = 0
dp_ratio = 0
q0 = dp_ratio * (2*np.pi/Lz)
K11 = 2.33
K22 = 2.33
K33 = 2.33
K24 = 0
B = 1.73e6
C = -2.12e6
A = -0.172e6
Sexp = (-B + np.sqrt(B**2 - 24*A*C))/(4*C)
L1 = (K33 - K11 + 3*K22)/(6 * Sexp**2)
L2 = (K11 - K22 - K24)/(Sexp**2)
L3 = K24/(Sexp**2)
L4 = q0 * K22 * 2/(Sexp**2) 

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
z = np.linspace(0, Lz, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

n0 = initial_director_field(X, Y, Z)
n = n0
Q = n2Q(n)
Qb = dirichlet_boundary_conditions(X, Y, Z)
Q[..., -1] = Qb[..., -1]
Q[..., 0] = Qb[..., 0]


Maxstep = 1001
alpha = 0.5*h*h/K11 

for iterstep in range(Maxstep):
    pseudo_grad = pseudo_grad_energy_elastic(Q)+ pseudo_grad_energy_LdG(Q)
    Q = Q - alpha * pseudo_grad
    Q[..., -1] = Qb[..., -1]
    Q[..., 0] = Qb[..., 0]

n, S0 = Q2n(Q)
nx, ny, nz = n[0], n[1], n[2]

























